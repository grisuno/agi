#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
AGI Voice Layer v2.0 - Capa de lenguaje robusta para sistema de expertos AGI
Corrección de problemas de routing y generación de respuestas
"""

import torch
import json
import time
import ollama
import re
from typing import Dict, Any, Tuple, Optional, List

# Importamos los componentes existentes de tu sistema AGI
try:
    from agi import (
        Grokkit,
        get_parity_dataset,
        generate_wave_data,
        generate_kepler_data,
        generate_and_save_chaotic_pendulum_dataset
    )
    from unificado import UnifiedGrokkitAgent
except ImportError as e:
    print(f"❌ Error importando módulos AGI: {e}")
    print("Asegúrate de tener los archivos agi.py y unificado.py en el mismo directorio")
    exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Dispositivo: {DEVICE}")

class AGIVoiceLayer:
    """Capa de lenguaje robusta que articula respuestas usando los expertos AGI"""
    
    def __init__(self, model_name: str = 'qwen2.5:0.5b', use_unified: bool = True, debug: bool = True):
        """Inicializa la capa de voz con el modelo de lenguaje y los expertos"""
        self.llm_model = model_name
        self.use_unified = use_unified
        self.debug = debug
        
        # Inicializamos el sistema de expertos AGI
        if use_unified:
            CASSETTES = {
                'parity': "weights/grok_model_stage4_n64_d1024_adaptive.pth",
                'wave': "weights/wave_grok_cnn_physics_cassette.pth", 
                'kepler': "weights/kepler_base_model.pth",
                'pendulum': "weights/symplectic_double_pendulum_grok_cassette.pth"
            }
            print("🧠 Inicializando UnifiedGrokkitAgent...")
            self.agent = UnifiedGrokkitAgent(CASSETTES)
        else:
            print("🧠 Inicializando Grokkit...")
            self.agent = Grokkit(load_weights=True)
        
        self.agent.eval()
        self.agent.to(DEVICE)
        
        # Definiciones claras de cada experto para el LLM
        self.expert_definitions = {
            'parity': {
                'description': 'Determina si la suma de los primeros k bits en un número binario es par o impar',
                'input_format': 'Número binario (string de 0s y 1s) y valor k (entero)',
                'example': 'bits="101010", k=3 → suma de 101 = 2 (par)'
            },
            'wave': {
                'description': 'Simula la propagación de ondas en medios físicos usando la ecuación de onda',
                'input_format': 'Tamaño de malla N (entero) y tipo de condición inicial',
                'example': 'N=32, initial_condition="gaussian" → simula onda gaussiana'
            },
            'kepler': {
                'description': 'Calcula posiciones orbitales según las leyes de Kepler',
                'input_format': 'Posición inicial (x0,y0), momento angular h, excentricidad e, tiempo t',
                'example': 'x0=1.0, y0=0.0, h=1.0, e=0.5, t=1.0 → posición orbital'
            },
            'pendulum': {
                'description': 'Simula el comportamiento caótico de un doble péndulo',
                'input_format': 'Ángulos iniciales q1,q2 y momentos p1,p2',
                'example': 'q1=0.1, q2=0.2, p1=0.0, p2=0.0 → estado inicial del péndulo'
            }
        }
        
        # Palabras clave para cada dominio (fallback robusto)
        self.domain_keywords = {
            'parity': ['bit', 'binario', 'par', 'impar', 'suma', 'binaria', '0', '1', 'número binario'],
            'wave': ['onda', 'wave', 'propagación', 'medio', 'física', 'simulación', 'amplitud', 'frecuencia'],
            'kepler': ['órbita', 'kepler', 'planeta', 'satélite', 'posición', 'tiempo', 'astronomía', 'celestial'],
            'pendulum': ['péndulo', 'chaos', 'caótico', 'ángulo', 'momento', 'dinámica', 'doble péndulo', 'estado']
        }
    
    def _extract_binary_number(self, text: str) -> str:
        """Extrae el número binario de una pregunta usando regex"""
        # Buscar patrones como "101010", "número 101010", "binario 101010"
        binary_pattern = r'\b[01]+\b'
        matches = re.findall(binary_pattern, text)
        
        if matches:
            return matches[0]  # Tomar el primer número binario encontrado
        
        # Si no encuentra, buscar después de palabras clave
        text_lower = text.lower()
        for keyword in ['numero', 'número', 'binario', 'bits']:
            if keyword in text_lower:
                parts = text_lower.split(keyword)
                if len(parts) > 1:
                    # Buscar números en la parte siguiente
                    next_part = parts[1]
                    matches = re.findall(r'\b[01]+\b', next_part)
                    if matches:
                        return matches[0]
        
        return "101010"  # Valor por defecto
    
    def _extract_k_value(self, text: str) -> int:
        """Extrae el valor k (número de bits a sumar)"""
        # Buscar patrones como "primeros 3 bits", "3 bits", "k=3"
        k_pattern = r'primeros?\s+(\d+)\s+bits?|k\s*=\s*(\d+)|(\d+)\s+bits?'
        matches = re.findall(k_pattern, text.lower())
        
        if matches:
            for match in matches[0]:
                if match:
                    try:
                        return int(match)
                    except:
                        continue
        
        # Palabras que implican paridad total
        if any(word in text.lower() for word in ['todos', 'total', 'completo']):
            binary_num = self._extract_binary_number(text)
            return len(binary_num)
        
        return 3  # Valor por defecto razonable
    
    def _domain_routing_heuristic(self, question: str) -> str:
        """Routing basado en heurísticas de palabras clave (más robusto que el LLM)"""
        question_lower = question.lower()
        
        # Primero: verificar si hay números binarios
        if re.search(r'\b[01]+\b', question_lower):
            return 'parity'
        
        # Segundo: buscar palabras clave específicas
        max_score = 0
        best_domain = 'parity'  # Default
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > max_score:
                max_score = score
                best_domain = domain
        
        # Si no hay coincidencias claras, usar heurística adicional
        if max_score == 0:
            if any(word in question_lower for word in ['número', 'numero', 'par', 'impar']):
                return 'parity'
            elif any(word in question_lower for word in ['posición', 'posición', 'orbita', 'órbita']):
                return 'kepler'
            elif any(word in question_lower for word in ['estado', 'angulo', 'ángulo', 'momento']):
                return 'pendulum'
            elif any(word in question_lower for word in ['onda', 'propaga', 'medio']):
                return 'wave'
        
        return best_domain
    
    def _prepare_input_for_expert(self, expert_name: str, parameters: Dict[str, Any], question: str) -> torch.Tensor:
        """Prepara los datos de entrada según el experto necesario, con fallbacks robustos"""
        try:
            if expert_name == 'parity':
                # Extraer número binario y k de la pregunta directamente
                bits = self._extract_binary_number(question)
                k = self._extract_k_value(question)
                
                # Asegurar que bits tenga al menos k caracteres
                if len(bits) < k:
                    bits = bits.ljust(k, '0')
                
                # Convertir a tensor (64 bits es el tamaño estándar)
                x = torch.zeros(1, 64)
                for i, bit in enumerate(bits[:64]):
                    x[0, i] = float(bit)
                
                if self.debug:
                    print(f"🔧 Parity - bits: '{bits}', k={k}")
                
                return x.to(DEVICE)
                
            elif expert_name == 'wave':
                N = parameters.get('N', 32)
                initial_condition = parameters.get('initial_condition', 'gaussian')
                
                wave_x, _, _, _, _ = generate_wave_data(N=N, T=1)
                if self.debug:
                    print(f"🔧 Wave - N={N}, condición: {initial_condition}")
                return wave_x[:1].to(DEVICE)
                
            elif expert_name == 'kepler':
                # Usar valores por defecto razonables
                x0 = parameters.get('x0', 1.0)
                y0 = parameters.get('y0', 0.0) 
                h = parameters.get('h', 1.0)
                e = parameters.get('e', 0.5)
                t = parameters.get('t', 1.0)
                
                x = torch.tensor([[x0, y0, h, e, t]], dtype=torch.float32)
                if self.debug:
                    print(f"🔧 Kepler - x0={x0}, y0={y0}, h={h}, e={e}, t={t}")
                return x.to(DEVICE)
                
            elif expert_name == 'pendulum':
                q1 = parameters.get('q1', 0.1)
                q2 = parameters.get('q2', 0.2)
                p1 = parameters.get('p1', 0.0)
                p2 = parameters.get('p2', 0.0)
                
                x = torch.tensor([[q1, q2, p1, p2]], dtype=torch.float32)
                if self.debug:
                    print(f"🔧 Pendulum - q1={q1}, q2={q2}, p1={p1}, p2={p2}")
                return x.to(DEVICE)
                
            else:
                if self.debug:
                    print(f"⚠️  Dominio desconocido '{expert_name}', usando parity como fallback")
                return self._prepare_input_for_expert('parity', parameters, question)
                
        except Exception as e:
            if self.debug:
                print(f"❌ Error preparando entrada: {e}")
                print("🔄 Usando valores por defecto seguros")
            
            # Valores por defecto seguros para cada dominio
            if expert_name == 'parity':
                return torch.zeros(1, 64).to(DEVICE)
            elif expert_name == 'wave':
                wave_x, _, _, _, _ = generate_wave_data(N=32, T=1)
                return wave_x[:1].to(DEVICE)
            elif expert_name == 'kepler':
                return torch.tensor([[1.0, 0.0, 1.0, 0.5, 1.0]], dtype=torch.float32).to(DEVICE)
            else:
                return torch.tensor([[0.1, 0.2, 0.0, 0.0]], dtype=torch.float32).to(DEVICE)
    
    def _interpret_technical_result(self, expert_name: str, result: torch.Tensor, 
                                   parameters: Dict[str, Any], question: str) -> str:
        """Convierte el resultado técnico en una descripción precisa en lenguaje natural"""
        try:
            result = result.cpu()  # Mover a CPU para operaciones
            
            if expert_name == 'parity':
                # Extraer bits y k de la pregunta original
                bits = self._extract_binary_number(question)
                k = self._extract_k_value(question)
                selected_bits = bits[:k]
                
                if result.dim() > 1:
                    prediction = torch.argmax(result, dim=1).item()
                else:
                    prediction = torch.argmax(result).item()
                
                sum_bits = sum(int(bit) for bit in selected_bits)
                is_even = sum_bits % 2 == 0
                actual_prediction = is_even
                
                if self.debug:
                    print(f"📊 Parity - bits: {bits}, k={k}, bits seleccionados: {selected_bits}")
                    print(f"📊 Parity - suma: {sum_bits}, ¿par?: {actual_prediction}, predicción modelo: {prediction == 0}")
                
                # Verificar consistencia
                if (prediction == 0) != actual_prediction:
                    warning = " ⚠️ (el modelo podría estar incorrecto)"
                else:
                    warning = ""
                
                return f"La suma de los primeros {k} bits ({selected_bits}) es {sum_bits}, que es {'par' if actual_prediction else 'impar'}.{warning}"
                
            elif expert_name == 'wave':
                # Calcular estadísticas básicas de la onda
                if result.dim() == 3:
                    wave_data = result[0, 0].numpy()  # Tomar primer canal
                else:
                    wave_data = result.numpy()
                
                amplitude = float(abs(wave_data).max())
                mean_val = float(wave_data.mean())
                
                return f"Onda simulada con amplitud máxima de {amplitude:.4f} y valor medio de {mean_val:.4f}."
                
            elif expert_name == 'kepler':
                if result.dim() > 1:
                    x_t, y_t = result[0, 0].item(), result[0, 1].item()
                else:
                    x_t, y_t = result[0].item(), result[1].item()
                
                t = parameters.get('t', 1.0)
                return f"Posición orbital en tiempo t={t}: x = {x_t:.4f}, y = {y_t:.4f}"
                
            elif expert_name == 'pendulum':
                if result.dim() > 1:
                    state = result[0].tolist()
                else:
                    state = result.tolist()
                
                q1, q2, p1, p2 = state[:4]
                return f"Estado del doble péndulo: q1 = {q1:.4f}, q2 = {q2:.4f}, p1 = {p1:.4f}, p2 = {p2:.4f}"
                
            else:
                return f"Resultado del experto {expert_name}: {result.tolist()}"
                
        except Exception as e:
            return f"Error interpretando resultado: {str(e)}"
    
    def _get_expert_analysis(self, question: str) -> Dict[str, Any]:
        """Análisis robusto con fallback a heurísticas si el LLM falla"""
        try:
            # Primero intentar con LLM
            prompt = f"""
            Analiza ESTA PREGUNTA ESPECÍFICA: "{question}"
            
            Determina EXACTAMENTE:
            1. Qué experto se necesita (SOLO: parity, wave, kepler, pendulum)
            2. Qué parámetros específicos necesita ESE experto
            3. Una explicación MUY BREVE de por qué
            
            REGLAS ESTRICTAS:
            - SI hay números binarios (0s y 1s) → SIEMPRE usar "parity"
            - SI menciona par/impar y números → SIEMPRE usar "parity"
            - NO inventar expertos que no existan
            - SER ESPECÍFICO con los parámetros
            - SI no estás seguro, usar "parity" como fallback
            
            Ejemplos correctos:
            - "este numero es par 101010" → expert_needed="parity", parameters={{"bits": "101010", "k": 3}}
            - "onda con 64 puntos" → expert_needed="wave", parameters={{"N": 64}}
            """
            
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                format='json',
                options={'temperature': 0.1, 'num_ctx': 2048}
            )
            
            response_text = response['response'].strip()
            
            # Limpiar respuesta JSON
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            analysis = json.loads(response_text)
            
            # Validar análisis
            expert = analysis.get('expert_needed', '').lower()
            if expert not in ['parity', 'wave', 'kepler', 'pendulum']:
                raise ValueError(f"Experto inválido: {expert}")
            
            return analysis
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  LLM falló ({str(e)}), usando heurísticas robustas")
            
            # Fallback a heurísticas robustas
            expert_name = self._domain_routing_heuristic(question)
            
            # Extraer parámetros específicos según el dominio
            parameters = {}
            
            if expert_name == 'parity':
                bits = self._extract_binary_number(question)
                k = self._extract_k_value(question)
                parameters = {"bits": bits, "k": k}
            elif expert_name == 'wave':
                # Buscar tamaño de malla en la pregunta
                n_match = re.search(r'(\d+)\s*puntos?|(\d+)\s*malla', question.lower())
                N = int(n_match.group(1)) if n_match else 32
                parameters = {"N": N}
            elif expert_name == 'kepler':
                # Parámetros por defecto razonables
                parameters = {"x0": 1.0, "y0": 0.0, "h": 1.0, "e": 0.5, "t": 1.0}
            elif expert_name == 'pendulum':
                parameters = {"q1": 0.1, "q2": 0.2, "p1": 0.0, "p2": 0.0}
            
            return {
                "expert_needed": expert_name,
                "parameters": parameters,
                "explanation": f"Seleccionado por heurísticas robustas: {expert_name}"
            }
    
    def _generate_final_response(self, question: str, expert_result: str, expert_name: str) -> str:
        """Genera la respuesta final natural basada en el resultado real del experto"""
        # Template específico para cada dominio
        domain_templates = {
            'parity': """Eres un asistente científico preciso. Responde SOLO a esta pregunta específica:

            Pregunta: "{question}"
            Resultado del análisis de paridad: {expert_result}

            Instrucciones:
            - Responde DIRECTAMENTE la pregunta SIN introducciones largas
            - SÉ BREVE pero completo
            - Usa el resultado técnico proporcionado, NO inventes
            - Si el resultado dice que es par, di que es par; si dice impar, di que es impar
            - NO menciones implementación técnica
            - Formato: "El número [binario] es [par/impar] porque la suma de los primeros [k] bits es [X]."

            Ejemplo correcto para "este numero es par 101010":
            "El número 101010 es par porque la suma de los primeros 3 bits (101) es 2, que es par."
            """,
            
            'wave': """Eres un físico explicando resultados de simulación. Responde SOLO a:

            Pregunta: "{question}"
            Resultado de simulación de onda: {expert_result}

            Instrucciones:
            - Explica qué significa el resultado en términos físicos simples
            - SÉ CONCISO
            - NO inventes detalles que no estén en el resultado
            - Ejemplo: "La onda tendrá una amplitud máxima de 0.5 unidades."
            """,
            
            'kepler': """Eres un astrónomo explicando posiciones orbitales. Responde SOLO a:

            Pregunta: "{question}"
            Resultado orbital: {expert_result}

            Instrucciones:
            - Da la posición en coordenadas x,y
            - Menciona el tiempo si es relevante
            - SÉ PRECISO con los números
            - Ejemplo: "En el tiempo t=1.0, el planeta estará en la posición x=0.8, y=0.6."
            """,
            
            'pendulum': """Eres un físico de sistemas caóticos. Responde SOLO a:

            Pregunta: "{question}"
            Resultado del péndulo: {expert_result}

            Instrucciones:
            - Describe el estado actual del sistema
            - SÉ ESPECÍFICO con los valores
            - Ejemplo: "El doble péndulo tiene ángulos de q1=0.1 rad y q2=0.2 rad con momentos p1=0.0 y p2=0.0."
            """
        }
        
        template = domain_templates.get(expert_name, domain_templates['parity'])
        prompt = template.format(question=question, expert_result=expert_result)
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={'temperature': 0.2, 'num_ctx': 2048}
            )
            
            response_text = response['response'].strip()
            
            # Post-procesamiento para eliminar introducciones innecesarias
            if expert_name == 'parity' and "El número" not in response_text and "es par" in response_text.lower():
                # Forzar formato correcto para paridad
                bits = self._extract_binary_number(question)
                k = self._extract_k_value(question)
                is_even = "par" in expert_result.lower()
                return f"El número {bits} {'es par' if is_even else 'es impar'} porque la suma de los primeros {k} bits es {'par' if is_even else 'impar'}."
            
            return response_text
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error generando respuesta final: {e}")
                print(f"🧠 Usando respuesta fallback para {expert_name}")
            
            # Respuesta fallback por dominio
            if expert_name == 'parity':
                bits = self._extract_binary_number(question)
                is_even = "par" in expert_result.lower()
                return f"{'✅' if is_even else '❌'} El número binario {bits} {'es par' if is_even else 'es impar'}."
            else:
                return f"Resultado del experto {expert_name}: {expert_result}"
    
    def respond_to_question(self, question: str) -> str:
        """Proceso completo robusto con múltiples fallbacks"""
        if not question.strip():
            return "❓ Por favor, formula una pregunta específica."
        
        print(f"\n{'='*60}")
        print(f"❓ Pregunta: {question}")
        print(f"{'='*60}")
        
        total_start_time = time.time()
        
        # Paso 1: Análisis de dominio (con fallbacks)
        analysis_start = time.time()
        analysis = self._get_expert_analysis(question)
        analysis_time = time.time() - analysis_start
        
        expert_name = analysis.get('expert_needed', 'parity').lower()
        parameters = analysis.get('parameters', {})
        explanation = analysis.get('explanation', 'Análisis automático')
        
        print(f"🔍 Análisis ({analysis_time:.2f}s):")
        print(f"   🧠 Experto seleccionado: {expert_name}")
        print(f"   ⚙️  Parámetros: {parameters}")
        print(f"   💡 Explicación: {explanation}")
        
        # Paso 2: Preparar entrada y obtener resultado del experto
        expert_start = time.time()
        input_tensor = self._prepare_input_for_expert(expert_name, parameters, question)
        
        with torch.no_grad():
            if self.use_unified:
                output, predicted_domain, confidence = self.agent(input_tensor)
                # Forzar el dominio correcto si el router se equivoca
                if predicted_domain != expert_name:
                    if self.debug:
                        print(f"⚠️  Router predijo '{predicted_domain}' pero necesitamos '{expert_name}'")
                    # Usar el experto correcto manualmente
                    cassette = getattr(self.agent, expert_name)
                    output = cassette(input_tensor)
            else:
                output, predicted_domain, probs = self.agent(input_tensor)
                if predicted_domain != expert_name:
                    if self.debug:
                        print(f"⚠️  Router predijo '{predicted_domain}' pero necesitamos '{expert_name}'")
                    cassette = getattr(self.agent, expert_name)
                    output = cassette(input_tensor)
        
        expert_time = time.time() - expert_start
        
        print(f"🧠 Expertos AGI ({expert_time:.2f}s):")
        print(f"   📈 Salida cruda: {output.shape if hasattr(output, 'shape') else 'escalar'}")
        
        # Paso 3: Interpretar resultado técnico
        interpretation_start = time.time()
        technical_result = self._interpret_technical_result(expert_name, output, parameters, question)
        interpretation_time = time.time() - interpretation_start
        
        print(f"📊 Interpretación ({interpretation_time:.2f}s):")
        print(f"   📝 {technical_result}")
        
        # Paso 4: Generar respuesta final natural
        response_start = time.time()
        final_response = self._generate_final_response(question, technical_result, expert_name)
        response_time = time.time() - response_start
        
        print(f"🗣️  Respuesta ({response_time:.2f}s):")
        print(f"   💬 {final_response}")
        
        # Tiempo total
        total_time = time.time() - total_start_time
        print(f"\n⏱️  Tiempo total: {total_time:.2f}s")
        
        return final_response
    
    def interactive_mode(self):
        """Modo interactivo mejorado con manejo de errores"""
        print("\n🚀 AGI Voice Layer v2.0 - Sistema robusto de expertos")
        print("💡 Escribe preguntas como:")
        print("   • 'este numero es par 101010'")
        print("   • 'onda con 64 puntos de malla'")
        print("   • 'posición orbital en t=2.0'")
        print("   • 'estado péndulo con q1=0.1, q2=0.2'")
        print("🚪 Escribe 'salir' para terminar\n")
        
        while True:
            try:
                question = input("\n🤔 Tu pregunta: ").strip()
                if question.lower() in ['salir', 'exit', 'quit', '']:
                    print("\n👋 ¡Hasta luego! AGI Voice Layer finalizado.")
                    break
                
                if not question:
                    continue
                
                response = self.respond_to_question(question)
                print(f"\n🤖 AGI: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Interrumpido por el usuario")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {str(e)}")
                print("💡 Intenta reformular tu pregunta o verifica que Ollama esté ejecutándose.")

def demo_voice_layer():
    """Demostración con ejemplos corregidos"""
    voice_layer = AGIVoiceLayer(model_name='qwen2.5:0.5b', use_unified=False, debug=True)
    
    ejemplos = [
        "este numero es par 101010",
        "¿es par la suma de los primeros 4 bits de 11100011?",
        "simula una onda con 64 puntos de malla",
        "calcula posición orbital con h=1.2, e=0.3, t=2.5",
        "estado de péndulo con ángulos 0.1 y 0.2 radianes"
    ]
    
    print("🧪 Demostración AGI Voice Layer v2.0")
    print("=" * 60)
    
    for i, pregunta in enumerate(ejemplos, 1):
        print(f"\n\n🔹 Ejemplo {i}/{len(ejemplos)}")
        voice_layer.respond_to_question(pregunta)
        time.sleep(0.5)  # Pausa para mejor legibilidad
    
    print("\n" + "="*60)
    print("✅ Demostración completada exitosamente")
    print("="*60)

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("🚀 Inicializando AGI Voice Layer v2.0...")
    
    # Verificar que Ollama esté disponible
    try:
        print("🔍 Verificando conexión con Ollama...")
        ollama.list()
        print("✅ Ollama disponible")
    except Exception as e:
        print(f"❌ Error conectando con Ollama: {e}")
        print("💡 Asegúrate de que Ollama esté ejecutándose y que el modelo 'qwen2.5:0.5b' esté descargado")
        print("   Comandos útiles:")
        print("   ollama serve")
        print("   ollama pull qwen2.5:0.5b")
        exit(1)
    
    # Ejecutar demostración o modo interactivo
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_voice_layer()
    else:
        voice_layer = AGIVoiceLayer(model_name='qwen2.5:0.5b', debug=True)
        voice_layer.interactive_mode()
