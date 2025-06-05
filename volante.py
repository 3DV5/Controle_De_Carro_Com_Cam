import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from math import atan2, degrees
import time

# Configurações do PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

# Inicialização do MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # Detecta ambas as mãos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Configurações da câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variáveis para suavização do volante
SMOOTHING_FACTOR = 5
angle_history = []
current_steering = 0

# Variáveis para controle de aceleração/freio
accel_active = False
brake_active = False
brake_start_time = 0
reverse_active = False

# Linha horizontal para direção (posição em 1/3 da altura da tela)
HORIZONTAL_LINE_Y_RATIO = 1/3
HORIZONTAL_LINE_X = 100  # Posição X da linha horizontal (lado esquerdo)
TOLERANCIA_PIXELS = 30  # Tolerância para zona neutra

def calculate_steering_angle(hand_landmarks, frame_width):
    """Calcula o ângulo de direção baseado na posição da mão"""
    # Pontos de referência (polegar e dedo médio)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Calcular ângulo entre os dedos
    dx = middle_tip.x - thumb_tip.x
    dy = middle_tip.y - thumb_tip.y
    angle = degrees(atan2(dy, dx))
    
    # Normalizar ângulo para -90 (esquerda) a +90 (direita)
    normalized_angle = np.interp(angle, [-45, 45], [-90, 90])
    
    # Calcular posição horizontal relativa
    hand_center_x = (thumb_tip.x + middle_tip.x) / 2
    screen_position = np.interp(hand_center_x, [0.2, 0.8], [-100, 100])
    
    # Combinar ângulo e posição
    combined_steering = (normalized_angle * 0.3 + screen_position * 0.7)
    
    return combined_steering

def is_hand_open(hand_landmarks):
    """Verifica se a mão está aberta (acelerar)"""
    # Pontas dos dedos: 8 (indicador), 12 (médio), 16 (anelar), 20 (mindinho)
    finger_tips = [8, 12, 16, 20]
    mcp_joints = [6, 10, 14, 18]  # Junta do meio (metacarpofalangeana)
    
    for tip, mcp in zip(finger_tips, mcp_joints):
        # Se a ponta do dedo estiver acima da junta MCP, o dedo está dobrado
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            return False
    return True

def is_hand_closed(hand_landmarks):
    """Verifica se a mão está fechada (freiar)"""
    finger_tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]  # Primeira junta (proximal)
    
    for tip, pip in zip(finger_tips, pip_joints):
        # Se a ponta do dedo estiver acima da junta PIP, o dedo não está completamente fechado
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            return False
    return True

def is_one_finger_up(hand_landmarks):
    """Verifica se apenas um dedo está levantado (acelerar)"""
    # Pontas dos dedos: 8 (indicador), 12 (médio), 16 (anelar), 20 (mindinho)
    finger_tips = [8, 12, 16, 20]
    mcp_joints = [6, 10, 14, 18]
    up_count = 0
    for tip, mcp in zip(finger_tips, mcp_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            up_count += 1
    return up_count == 1

def apply_smoothing(new_angle):
    """Aplica suavização ao ângulo de direção"""
    global angle_history
    
    angle_history.append(new_angle)
    if len(angle_history) > SMOOTHING_FACTOR:
        angle_history.pop(0)
    
    return sum(angle_history) / len(angle_history)

def send_steering_command(angle):
    """Envia comandos para o jogo baseado no ângulo de direção"""
    global current_steering
    
    # Limitar ângulo
    angle = max(-100, min(100, angle))
    
    # Só enviar comando se houver mudança significativa
    if abs(angle - current_steering) > 5:
        if angle < -20:  # Virar à esquerda
            pyautogui.keyDown('a')
            pyautogui.keyUp('d')
        elif angle > 20:  # Virar à direita
            pyautogui.keyDown('d')
            pyautogui.keyUp('a')
        else:  # Centro
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
        
        current_steering = angle

def send_steering_command_by_hand_y(hand_landmarks, frame):
    """Envia comandos de direção baseado na posição vertical da mão esquerda"""
    global current_steering
    # Pega o centro da mão (média do polegar e dedo médio)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    hand_center_y = int(((thumb_tip.y + middle_tip.y) / 2) * frame.shape[0])
    horizontal_line_y = int(frame.shape[0] * HORIZONTAL_LINE_Y_RATIO)
    # Decide direção
    if hand_center_y < horizontal_line_y - TOLERANCIA_PIXELS:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
        current_steering = -100
    elif hand_center_y > horizontal_line_y + TOLERANCIA_PIXELS:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
        current_steering = 100
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        current_steering = 0
    # Mostra info na tela
    cv2.putText(frame, f"Y mao: {hand_center_y} / Linha: {horizontal_line_y}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

def update_accel_brake(accel, brake):
    """Atualiza o estado de aceleração e freio"""
    global accel_active, brake_active, brake_start_time, reverse_active
    
    # Atualizar aceleração
    if accel and not accel_active:
        pyautogui.keyDown('w')
        accel_active = True
    elif not accel and accel_active:
        pyautogui.keyUp('w')
        accel_active = False
    
    # Atualizar freio
    if brake and not brake_active:
        pyautogui.keyDown('s')
        brake_active = True
        brake_start_time = time.time()
        reverse_active = False
    elif not brake and brake_active:
        pyautogui.keyUp('s')
        brake_active = False
        reverse_active = False
    
    # Verificar se deve ativar ré (freio pressionado por mais de 20 segundos)
    if brake_active and not reverse_active:
        if time.time() - brake_start_time > 20:
            reverse_active = True
            # Engatar a ré (pressionar R uma vez)
            pyautogui.press('r')

# Loop principal
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Espelhar a imagem para comportamento mais intuitivo
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detecção de mãos
    results = hands.process(frame_rgb)
    
    # Inicializar estados
    left_hand_detected = False
    right_hand_detected = False
    steering_angle = 0
    hand_open = False
    hand_closed = False
    
    if results.multi_hand_landmarks:
        # Identificar mãos e seus landmarks
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determinar se é mão esquerda ou direita
            handedness = results.multi_handedness[hand_idx].classification[0].label
            
            # Desenhar landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
            
            if handedness == 'Left':
                # Mão esquerda: controle de direção
                left_hand_detected = True
                left_hand_closed = is_hand_closed(hand_landmarks)
                if left_hand_closed:
                    pyautogui.keyUp('a')
                    pyautogui.keyUp('d')
                    current_steering = 0
                    cv2.putText(frame, f"Direcao: Centro (mao fechada)", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    send_steering_command_by_hand_y(hand_landmarks, frame)
                    cv2.putText(frame, f"Direcao: {'Esquerda' if current_steering < 0 else 'Direita' if current_steering > 0 else 'Centro'}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            elif handedness == 'Right':
                # Mão direita: controle de aceleração/freio
                right_hand_detected = True
                hand_closed = is_hand_closed(hand_landmarks)
                one_finger_up = is_one_finger_up(hand_landmarks)
                # Atualizar estados de aceleração/freio
                update_accel_brake(one_finger_up, hand_closed)
                # Mostrar estado na tela
                if hand_closed:
                    status = "Freando/Re"
                elif one_finger_up:
                    status = "Acelerando (1 dedo)"
                else:
                    status = "Neutro"
                cv2.putText(frame, f"Mão Direita: {status}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Se nenhuma mão esquerda for detectada, soltar teclas de direção
    if not left_hand_detected:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
    
    # Se nenhuma mão direita for detectada, soltar teclas de aceleração/freio
    if not right_hand_detected:
        update_accel_brake(False, False)
    
    # Mostrar tempo de freio se estiver ativo
    if brake_active:
        brake_time = time.time() - brake_start_time
        cv2.putText(frame, f"Freio: {brake_time:.1f}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if reverse_active:
            cv2.putText(frame, "REVERSE ATIVADA!", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Desenhar linha de referência para direção
    cv2.line(frame, (frame.shape[1]//2, 0), (frame.shape[1]//2, frame.shape[0]), (0, 255, 0), 2)
    # Desenhar linha horizontal do lado esquerdo
    horizontal_line_y = int(frame.shape[0] * HORIZONTAL_LINE_Y_RATIO)
    cv2.line(frame, (HORIZONTAL_LINE_X, horizontal_line_y), (frame.shape[1]//2, horizontal_line_y), (255, 0, 0), 2)
    
    # Mostrar instruções na tela
    cv2.putText(frame, "Mao Esquerda: Direcao", (frame.shape[1]-300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    cv2.putText(frame, "Mao Direita Aberta: Acelerar", (frame.shape[1]-300, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Mao Direita Fechada: Freiar", (frame.shape[1]-300, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Freio >20s: Re", (frame.shape[1]-300, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    
    # Mostrar frame
    cv2.imshow('Volante Virtual Completo - Forza Horizon 5', frame)
    
    # Sair com 'q' ou ESC
    key = cv2.waitKey(1)
    if key in (27, ord('q')):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()

# Garantir que todas as teclas sejam soltas ao sair
pyautogui.keyUp('a')
pyautogui.keyUp('d')
pyautogui.keyUp('w')
pyautogui.keyUp('s')