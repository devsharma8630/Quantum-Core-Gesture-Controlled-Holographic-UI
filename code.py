import cv2, mediapipe as mp, numpy as np, math, random, time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

class Particle:
    def __init__(self):
        self.angle = random.uniform(0,6.28)
        self.dist = random.randint(20,120)
        self.speed = random.uniform(0.01,0.06)

particles = [Particle() for _ in range(160)]

waves = []
hue = 0

while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    canvas = np.zeros((h,w,3),dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    cam = cv2.resize(frame,(200,150))
    canvas[10:160,10:210] = cam

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark

        # Check if hand is open
        open_fingers = 0
        tips = [8,12,16,20]
        bases = [6,10,14,18]
        for i in range(4):
            if lm[tips[i]].y < lm[bases[i]].y:
                open_fingers += 1

        hand_open = open_fingers >= 4

        # Anchor hologram on finger or palm
        fx, fy = int(lm[8].x*w), int(lm[8].y*h)
        bx, by = int(lm[6].x*w), int(lm[6].y*h)
        index_open = fy < by - 20

        if index_open:
            cx, cy = fx, fy
        else:
            cx, cy = int(lm[0].x*w), int(lm[0].y*h)

        ix,iy = int(lm[8].x*w),int(lm[8].y*h)
        tx,ty = int(lm[4].x*w),int(lm[4].y*h)
        size = int(math.hypot(ix-tx,iy-ty))
        size = max(50,min(size*3,220))

        mid = lm[9]; wrist = lm[0]
        angle = math.degrees(math.atan2(mid.y-wrist.y, mid.x-wrist.x))

        # Trigger waves ONLY when hand is open
        if hand_open and len(waves) < 6:
            waves.append(0)

        hue = (hue + 2) % 180
        hsv = np.uint8([[[hue,255,255]]])
        color = tuple(map(int,cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0][0]))

        for p in particles:
            p.angle += p.speed + angle*0.0005
            r = size + p.dist
            x = int(cx + math.cos(p.angle)*r)
            y = int(cy + math.sin(p.angle)*r)
            cv2.circle(canvas,(x,y),2,color,-1)
            cv2.line(canvas,(cx,cy),(x,y),color,1)

        cv2.circle(canvas,(cx,cy),size,color,3)

        for i in range(len(waves)):
            waves[i] += 12
            wave_hue = (hue + i*30) % 180
            hsv2 = np.uint8([[[wave_hue,255,255]]])
            wave_color = tuple(map(int,cv2.cvtColor(hsv2,cv2.COLOR_HSV2BGR)[0][0]))
            cv2.circle(canvas,(cx,cy),waves[i],wave_color,2)

        waves[:] = [w for w in waves if w < 400]

        # Hand skeleton
        for c in mp.solutions.hands.HAND_CONNECTIONS:
            p1 = lm[c[0]]; p2 = lm[c[1]]
            cv2.line(canvas,(int(p1.x*w),int(p1.y*h)),
                     (int(p2.x*w),int(p2.y*h)),(0,255,255),2)

        cv2.putText(canvas,"POWER OF PYTHON",(cx-110,cy-size-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv2.imshow("POWER PF PYTHON",canvas)
    if cv2.waitKey(1)==27: break

cap.release()
cv2.destroyAllWindows()
