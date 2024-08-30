# -*- coding: utf-8 -*-
'''
kernal v1.0
'''
import numpy as np
from scipy.stats import norm

class bullet(object):
    def __init__(self, center, angle, speed, owner):
        self.center = center.copy()
        self.speed = speed
        self.angle = angle
        self.owner = owner

class state(object):
    def __init__(self, time, agents, compet_info, done=False, detect=None, vision=None, atk=None):
        self.time = time
        self.agents = agents
        self.compet = compet_info
        self.done = done
        self.detect = detect
        self.vision = vision
        self.atk = atk

class record(object):
    def __init__(self, time, cars, compet_info, detect, vision, atk, bullets):
        self.time = time
        self.cars = cars
        self.compet_info = compet_info
        self.detect = detect
        self.vision = vision
        self.atk = atk
        self.bullets = bullets

class g_map(object):
    def __init__(self, length, width, areas, barriers):
        self.length = length
        self.width = width
        self.areas = areas
        self.barriers = barriers

class record_player(object):
    def __init__(self):
        self.map_length = 1200
        self.map_width = 800
        global pygame
        import pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.map_length, self.map_width))
        pygame.display.set_caption('RMUL-SentryDecisionEnv-Simulator')
        self.gray = (180, 180, 180)
        self.red = (190, 20, 20)
        self.blue = (10, 125, 181)
        self.areas = np.array([[[500.0, 700.0, 300.0, 500.0],
                                [0.0, 100.0, 0.0, 200.0],
                                [260.0, 350.0, 400.0, 495.0]],
                               [[500.0, 700.0, 300.0, 500.0],
                                [1100.0, 1200.0, 600.0, 800.0],
                                [850.0, 940.0, 300.0, 395.0]]], dtype='float32')
        self.barriers = np.array([[0.0, 195.0, 460.0, 505.5],
                                  [295.0, 340.0, 605.0, 800.0],
                                  [443.0, 473.0, 213.0, 510.0],
                                  [728.0, 772.0, 288.0, 585.0],
                                  [860.0, 907.0, 0.0, 192.0],
                                  [1007.0, 1200.0, 338.0, 375.0]], dtype='float32')
        
        # 为障碍物定义不同的图像
        barrier_image_paths = ['./imgs/redBaseUpWall.png', './imgs/redBaseDownWall.png',
                               './imgs/buffLeftWall.png', './imgs/buffRightWall.png',
                               './imgs/blueBaseUpWall.png', './imgs/blueDaseDownWall.png']
        
        # load barriers imgs
        self.barriers_img = []
        self.barriers_rect = []
        for i in range(self.barriers.shape[0]):
            self.barriers_img.append(pygame.image.load(barrier_image_paths[i]))
            self.barriers_rect.append(self.barriers_img[-1].get_rect())
            self.barriers_rect[-1].center = [self.barriers[i][0:2].mean(), self.barriers[i][2:4].mean()]
        # load areas imgs
        self.areas_img = []
        self.areas_rect = []
        for oi, o in enumerate(['red', 'blue']):
            for ti, t in enumerate(['bonus', 'supply', 'start']):
                self.areas_img.append(pygame.image.load('./imgs/area_{}_{}.png'.format(t, o)))
                self.areas_rect.append(self.areas_img[-1].get_rect())
                self.areas_rect[-1].center = [self.areas[oi, ti][0:2].mean(), self.areas[oi, ti][2:4].mean()]
        self.background = pygame.image.load('./imgs/background.png')
        self.chassis_img = pygame.image.load('./imgs/chassis_g.png')
        self.gimbal_img = pygame.image.load('./imgs/gimbal_g.png')
        self.bullet_img = pygame.image.load('./imgs/bullet_s.png')
        self.info_bar_img = pygame.image.load('./imgs/info_bar.png')
        self.bullet_rect = self.bullet_img.get_rect()
        self.info_bar_rect = self.info_bar_img.get_rect()
        self.info_bar_rect.center = [200, self.map_width/2]
        pygame.font.init()
        self.font = pygame.font.SysFont('info', 20)
        self.clock = pygame.time.Clock()

    def play(self, file):
        self.memory = np.load(file)
        i = 0
        stop = False
        flag = 0
        while True:
            self.time = self.memory[i].time
            self.cars = self.memory[i].cars
            self.car_num = len(self.cars)
            self.compet_info = self.memory[i].compet_info
            self.detect = self.memory[i].detect
            self.vision = self.memory[i].vision
            self.atk = self.memory[i].atk
            self.bullets = self.memory[i].bullets
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_TAB]: self.dev = True
            else: self.dev = False
            self.one_epoch()
            if pressed[pygame.K_SPACE] and not flag:
                flag = 50
                stop = not stop
            if flag > 0: flag -= 1
            if pressed[pygame.K_LEFT] and i > 10: i -= 10
            if pressed[pygame.K_RIGHT] and i < len(self.memory)-10: i += 10
            if i < len(self.memory)-1 and not stop: i += 1
            self.clock.tick(200)

    def one_epoch(self):
        self.screen.fill(self.gray)
        for i in range(len(self.barriers_rect)):
            self.screen.blit(self.barriers_img[i], self.barriers_rect[i])
        for i in range(len(self.areas_rect)):
            self.screen.blit(self.areas_img[i], self.areas_rect[i])
        for i in range(len(self.bullets)):
            self.bullet_rect.center = self.bullets[i].center
            self.screen.blit(self.bullet_img, self.bullet_rect)
        for n in range(self.car_num):
            chassis_rotate = pygame.transform.rotate(self.chassis_img, -self.cars[n, 3]-90)
            gimbal_rotate = pygame.transform.rotate(self.gimbal_img, -self.cars[n, 4]-self.cars[n, 3]-90)
            chassis_rotate_rect = chassis_rotate.get_rect()
            gimbal_rotate_rect = gimbal_rotate.get_rect()
            chassis_rotate_rect.center = self.cars[n, 1:3]
            gimbal_rotate_rect.center = self.cars[n, 1:3]
            self.screen.blit(chassis_rotate, chassis_rotate_rect)
            self.screen.blit(gimbal_rotate, gimbal_rotate_rect)
            select = np.where((self.vision[n] == 1))[0]+1
            select2 = np.where((self.detect[n] == 1))[0]+1
            info = self.font.render('{} | {}: {} {}'.format(int(self.cars[n, 6]), n+1, select, select2), True, self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3]+[-20, -60])
            info = self.font.render('{} {}'.format(int(self.cars[n, 10]), int(self.cars[n, 5])), True, self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3]+[-20, -45])
        info = self.font.render('time: {}'.format(self.time), False, (0, 0, 0))
        self.screen.blit(info, (8, 8))
        if self.dev:
            for n in range(self.car_num):
                wheels = self.check_points_wheel(self.cars[n])
                for w in wheels:
                    pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, w.astype(int), 3)
                armors = self.check_points_armor(self.cars[n])
                for a in armors:
                    pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, a.astype(int), 3)
            self.screen.blit(self.info_bar_img, self.info_bar_rect)
            for n in range(self.car_num):
                tags = ['owner', 'x', 'y', 'angle', 'yaw', 'heat', 'hp', 'freeze_time', 'is_supply', 
                        'can_shoot', 'bullet', 'stay_time', 'wheel_hit', 'armor_hit', 'car_hit']
                info = self.font.render('car {}'.format(n), False, (0, 0, 0))
                self.screen.blit(info, (8+n*100, 100))
                for i in range(self.cars[n].size):
                    info = self.font.render('{}: {}'.format(tags[i], int(self.cars[n, i])), False, (0, 0, 0))
                    self.screen.blit(info, (8+n*100, 117+i*17))
            info = self.font.render('red   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[0, 0], \
                                    self.compet_info[0, 1], self.compet_info[0, 2]), False, (0, 0, 0))
            self.screen.blit(info, (8, 372))
            info = self.font.render('blue   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[1, 0], \
                                self.compet_info[1, 1], self.compet_info[1, 2]), False, (0, 0, 0))
            self.screen.blit(info, (8, 389))
            self.screen.blit(self.background, (1400, 750))
        pygame.display.flip()

    def check_points_wheel(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-22.5, -29], [22.5, -29], 
                       [-22.5, -14], [22.5, -14], 
                       [-22.5, 14], [22.5, 14],
                       [-22.5, 29], [22.5, 29]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]

    def check_points_armor(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-6.5, -30], [6.5, -30], 
             [-18.5,  -7], [18.5,  -7],
             [-18.5,  0], [18.5,  0],
             [-18.5,  6], [18.5,  6],
             [-6.5, 30], [6.5, 30]])
        return [np.matmul(x, rotate_matrix) + car[1:3] for x in xs]

class kernal(object):
    def __init__(self, car_num, render=False, record=True):
        self.car_num = car_num
        self.render = render
        # below are params that can be challenged depended on situation
        self.bullet_speed = 12.5
        self.motion = 6
        self.rotate_motion = 4
        self.yaw_motion = 1
        self.camera_angle = 360 / 2
        self.lidar_angle = 360 / 2
        self.move_discount = 0.6
        # above are params that can be challenged depended on situation
        self.map_length = 2800     #单位是cm
        self.map_width = 1500
        self.theta = np.rad2deg(np.arctan(45/60))
        self.record=record
        self.areas = np.array([[[500.0, 700.0, 300.0, 500.0],
                                [0.0, 100.0, 0.0, 200.0],
                                [260.0, 350.0, 400.0, 495.0]],
                               [[500.0, 700.0, 300.0, 500.0],
                                [1100.0, 1200.0, 600.0, 800.0],
                                [850.0, 940.0, 300.0, 395.0]]], dtype='float32')
        self.barriers = np.array([[0.0, 195.0, 460.0, 505.5],        #添加了障碍物的范围，xmin,xmax,ymin,ymax
                                  [295.0, 340.0, 605.0, 800.0],
                                  [443.0, 473.0, 213.0, 510.0],
                                  [728.0, 772.0, 288.0, 585.0],
                                  [860.0, 907.0, 0.0, 192.0],
                                  [1007.0, 1200.0, 338.0, 375.0]], dtype='float32')
        self.region = [[[0,0],[0,480],[150,480],[150,350],[275,350],[280,460],[300,460],[300,0],[0,0]],
                                [[2800,1500],[2800,1020],[2650,1020],[2650,1150],[2525,1150],[2520,1040],[2500,1040],[2500,1500],[2800,1500]],
                                [[480,460],[530,460],[760,110],[720,110],[480,460]],
                                [[2320,1040],[2270,1040],[2040,1390],[2080,1390],[2320,1040]],
                                [[100,690],[100,700],[80,710],[80,770],[100,780],[170,835],[177,827],[190,934],[240,805],[240,785],[255,775],[255,700],[245,695],[245,680],[190,650],[175,655],[165,645],[100,690]],
                                [[2700,810],[2700,800],[2720,790],[2720,730],[2700,720],[2630,665],[2623,673],[2610,566],[2560,695],[2560,715],[2545,725],[2545,800],[2555,805],[2555,820],[2610,850],[2625,845],[2635,855],[2700,810]],
                                [[1180,105],[990,380],[950,390],[900,457],[905,500],[865,560],[865,815],[1025,1035],[1035,1027],[880,812],[880,565],[1205,105],[1180,105]],
                                [[1620,1395],[1810,1120],[1850,1110],[1900,1043],[1895,1000],[1935,940],[1935,685],[1775,465],[1765,473],[1920,688],[1920,935],[1595,1395],[1620,1395]],
                                [[1310,225],[1035,615],[1035,770],[1190,990],[1230,965],[1085,750],[1085,630],[1350,255],[1310,225]],
                                [[1490,1275],[1765,885],[1765,730],[1610,510],[1570,535],[1715,750],[1715,870],[1450,1245],[1490,1275]],
                                [[1120,1220],[1085,1240],[1085,1285],[1120,1305],[1155,1285],[1155,1245],[1120,1220]],
                                [[1680,280],[1715,260],[1715,215],[1680,195],[1645,215],[1645,255],[1680,280]],
                                [[0,1075],[115,1075],[115,1257],[145,1285],[140,1460],[120,1500],[0,1500],[0,1075]],
                                [[2800,425],[2685,425],[2685,243],[2655,215],[2660,40],[2680,0],[2800,0],[2800,425]],
                                [[305,1100],[305,1500],[325,1500],[330,1140],[320,1100],[305,1100]],
                                [[2495,400],[2495,0],[2475,0],[2470,360],[2480,400],[2495,400]],
                                [[465,1330],[450,1057],[480,1025],[735,1025],[860,1223],[940,1223],[940,1380],[600,1380],[600,1360],[920,1360],[920,1240],[600,1240],[600,1220],[810,1220],[710,1065],[500,1065],[485,1080],[485,1330],[465,1330]],
                                [[2335,170],[2350,443],[2320,475],[2065,475],[1940,277],[1860,277],[1860,120],[2200,120],[2200,140],[1880,140],[1880,260],[2200,260],[2200,280],[1990,280],[2090,435],[2300,435],[2315,420],[2315,170],[2335,170]],
                                [[1115,1375],[1250,1375],[1250,1395],[1115,1395],[1115,1375]],
                                [[1685,125],[1550,125],[1550,105],[1685,105],[1685,125]],
                                [[1250,1475],[1115,1475],[1115,1490],[1250,1500],[1250,1475]],
                                [[1550,25],[1685,25],[1685,10],[1550,0],[1550,25]],
                                [[1275,680],[1470,870],[1520,815],[1330,620],[1275,680]]]
        
        self.bases  = np.array([[1,1050, 150,0,1500,1500,1], 
                          [0,150, 650,0,1500,1500,1]], dtype='float32')
        
         # 为障碍物定义不同的图像
        barrier_image_paths = ['./imgs/redBaseUpWall.png', './imgs/redBaseDownWall.png',
                               './imgs/buffLeftWall.png', './imgs/buffRightWall.png',
                               './imgs/blueBaseUpWall.png', './imgs/blueDaseDownWall.png']
        if render:
            global pygame
            import pygame
            global time
            import time
            pygame.init()
            self.frequency = 0.1
            self.start_time = time.time()
            self.current_time = time.time()
            self.screen = pygame.display.set_mode((self.map_length, self.map_width))
            pygame.display.set_caption('RMUL SentryDecisionEnv Simulator')
            self.gray = (180, 180, 180)
            self.red = (190, 20, 20)
            self.blue = (10, 125, 181)
            # load barriers imgs
            self.barriers_img = []
            self.barriers_rect = []
            for i in range(self.barriers.shape[0]):
                self.barriers_img.append(pygame.image.load(barrier_image_paths[i]))
                self.barriers_rect.append(self.barriers_img[-1].get_rect())
                self.barriers_rect[-1].center = [self.barriers[i][0:2].mean(), self.barriers[i][2:4].mean()]
            # load areas imgs
            self.areas_img = []
            self.areas_rect = []
            for oi, o in enumerate(['red', 'blue']):
                # for ti, t in enumerate(['bonus', 'supply', 'start', 'start']):
                for ti, t in enumerate(['bonus', 'supply', 'start']):
                    self.areas_img.append(pygame.image.load('./imgs/area_{}_{}.png'.format(t, o)))
                    self.areas_rect.append(self.areas_img[-1].get_rect())
                    self.areas_rect[-1].center = [self.areas[oi, ti][0:2].mean(), self.areas[oi, ti][2:4].mean()]

            #加载基地贴图及参数
            self.base_blue_img = pygame.image.load('./imgs/blueBase.png')
            self.base_red_img = pygame.image.load('./imgs/redBase.png')
            self.base_blue_rect = self.base_blue_img.get_rect()
            self.base_red_rect = self.base_red_img.get_rect()
            self.base_blue_rect.center = [self.bases[0][1],self.bases[0][2]]
            self.base_red_rect.center = [self.bases[1][1],self.bases[1][2]]

            self.chassis_img = pygame.image.load('./imgs/chassis_g.png')
            self.gimbal_img = pygame.image.load('./imgs/gimbal_g.png')
            self.bullet_img = pygame.image.load('./imgs/bullet_s.png')
            self.info_bar_img = pygame.image.load('./imgs/info_bar.png')
            
            self.bullet_rect = self.bullet_img.get_rect()
            self.info_bar_rect = self.info_bar_img.get_rect()
            self.info_bar_rect.center = [200, self.map_width/2]
            self.background = pygame.image.load('./imgs/background.png')
            pygame.font.init()
            self.font = pygame.font.SysFont('info', 20)
            self.clock = pygame.time.Clock()

    def reset(self):
        self.time = 180
        self.orders = np.zeros((4, 8), dtype='int8')
        self.acts = np.zeros((self.car_num, 8),dtype='float32')
        self.obs = np.zeros((self.car_num, 17), dtype='float32')

        #赛事信息要增加跟多适配3v3的信息，比如基地血量，占领中心增益点进度，哨兵剩余可用补血量
        self.compet_info = np.array([[2, 1, 0, 0], [2, 1, 0, 0]], dtype='int16')
        self.atk = np.zeros((self.car_num, self.car_num ), dtype='int8')
        self.vision = np.zeros((self.car_num, self.car_num ), dtype='int8')
        self.detect = np.zeros((self.car_num, self.car_num ), dtype='int8')
        self.bullets = []
        self.epoch = 0
        self.n = 0
        self.dev = False
        self.memory = []
        #第一个是蓝方哨兵机器人 第二个是红方哨兵机器人 第三个是蓝方步兵机器人 第四个是红方步兵机器人
        #可能会影响原有demo中对2v1情况的使用因为对机器人的顺序进行了更改
        # cars = np.array([[1, 895, 347, 45, 0, 0, 500, 0, 0, 1, 0, 0, 0, 0, 0],
        #                  [0, 305, 447, -45, 0, 0, 500, 0, 0, 1, 0, 0, 0, 0, 0],
        #                  [1, 953, 142, 0, 0, 0, 200, 0, 0, 1, 0, 0, 0, 0, 0],
        #                  [0, 248, 652, 0, 0, 0, 200, 0, 0, 1, 0, 0, 0, 0, 0]], dtype='float32')
        cars = np.array([[1, 895, 347, 45, 0, 0, 500, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 500],
                         [0, 305, 447, -45, 0, 0, 500, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 500],
                         [0, 248, 652, 0, 0, 0, 200, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 200],
                         [1, 953, 142, 0, 0, 0, 200, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 200]], dtype='float32')
        #第一个是蓝方基地，第二个是红方基地，参数信息依次是：所属方、位置x、位置y角度、虚拟护盾血量、实际血量、是否无敌
        # self.bases  = np.array([[1,1050, 150,0,1500,1500,1], 
        #                   [0,150, 650,0,1500,1500,1]], dtype='float32')

        self.cars = cars[0:self.car_num]#+2指的是红蓝方的基地
        return state(self.time, self.cars, self.compet_info, self.time <= 0)

    def play(self):
        # human play mode, only when render == True
        assert self.render, 'human play mode, only when render == True'
        while True:
            if not self.epoch % 10:
                if self.get_order():
                    break
            self.one_epoch()

    def step(self, orders):
        # 创建一个长度为8的全零数组
        arr = np.zeros(8)
        # 将索引为orders-1的元素设置为1
        if orders==1:
            arr[0] = 1
        if orders==2:
            arr[0] = -1
        if orders==3:
            arr[1] = 1
        if orders==4:
            arr[1] = -1
        if orders==5:
            arr[2] = 1
        if orders==6:
            arr[2] = -1
        # arr[orders] = 1
        self.orders[0:self.car_num] = arr
        # self.orders = np.array(orders)
        done=0
        if self.time<120:
            done = 1
        for _ in range(10):
            self.one_epoch()
        return state(self.time, self.cars, self.compet_info, done, self.detect, self.vision)

    def one_epoch(self):
        for n in range(self.car_num):
            if not self.epoch % 10:
                self.orders_to_acts(n)
            # move car one by one
            self.move_car(n)
            if not self.epoch % 20:
                if self.cars[n, 5] >= 720:
                    self.cars[n, 6] -= (self.cars[n, 5] - 720) * 40
                    self.cars[n, 5] = 720
                elif self.cars[n, 5] > 360:
                    self.cars[n, 6] -= (self.cars[n, 5] - 360) * 4
                self.cars[n, 5] -= 12 if self.cars[n, 6] >= 400 else 24
            if self.cars[n, 5] <= 0: self.cars[n, 5] = 0
            if self.cars[n, 6] <= 0: self.cars[n, 6] = 0
            if not self.acts[n, 5]: self.acts[n, 4] = 0
        if not self.epoch % 200:
                self.time -= 1
                if not self.time % 60:
                    # self.compet_info[:, 0:3] = [2, 1, 0]
                    self.compet_info[:, 0:4] = [2, 1, 0, 0]
        self.get_camera_vision()
        self.get_lidar_vision()
        self.stay_check()
        self.current_time = time.time()
        if self.current_time - self.start_time >= self.frequency:

            self.check_hit()
            self.start_time = self.current_time
        # move bullet one by one
        i = 0
        while len(self.bullets):
            if self.move_bullet(i):
                del self.bullets[i]
                i -= 1
            i += 1
            if i >= len(self.bullets): break
        self.epoch += 1
        bullets = []
        for i in range(len(self.bullets)):
            bullets.append(bullet(self.bullets[i].center, self.bullets[i].angle, self.bullets[i].speed, self.bullets[i].owner))
        if self.record: self.memory.append(record(self.time, self.cars.copy(), self.compet_info.copy(), self.detect.copy(), self.vision.copy(), self.atk.copy(),bullets))
        if self.render: self.update_display()

    def move_car(self, n):
        if not self.cars[n, 7]:
            # move chassis
            if self.acts[n, 0]:
                p = self.cars[n, 3]
                self.cars[n, 3] += self.acts[n, 0]
                if self.cars[n, 3] > 180: self.cars[n, 3] -= 360
                if self.cars[n, 3] < -180: self.cars[n, 3] += 360
                if self.check_interface(n):
                    self.acts[n, 0] = -self.acts[n, 0] * self.move_discount
                    self.cars[n, 3] = p
            # move gimbal
            if self.acts[n, 1]:
                self.cars[n, 4] += self.acts[n, 1]
                if self.cars[n, 4] >= 180: self.cars[n, 4] -= 360
                if self.cars[n, 4] < -180: self.cars[n, 4] += 360  
            # print(self.acts[n, 7])
            if self.acts[n, 7]:
                if self.car_num > 1:
                    select = np.where((self.vision[n] == 1))[0]
                    if select.size:
                        angles = np.zeros(select.size)
                        for ii, i in enumerate(select):
                            if self.cars[i, 0] == self.cars[n, 0]:
                                continue
                            x, y = self.cars[i, 1:3] - self.cars[n, 1:3]
                            angle = np.angle(x+y*1j, deg=True) - self.cars[i, 3]
                            if angle >= 180: angle -= 360
                            if angle <= -180: angle += 360
                            if i < self.car_num:
                                if angle >= -self.theta and angle < self.theta:
                                    armor = self.get_armor(self.cars[i], 2)
                                elif angle >= self.theta and angle < 180-self.theta:
                                    armor = self.get_armor(self.cars[i], 3)
                                elif angle >= -180+self.theta and angle < -self.theta:
                                    armor = self.get_armor(self.cars[i], 1)
                                else: armor = self.get_armor(self.cars[i], 0)
                            else:
                                if angle >= -self.theta and angle < self.theta:
                                    armor = self.get_bases_armor(self.cars[i], 2)
                                elif angle >= self.theta and angle < 180-self.theta:
                                    armor = self.get_bases_armor(self.cars[i], 3)
                                elif angle >= -180+self.theta and angle < -self.theta:
                                    armor = self.get_bases_armor(self.cars[i], 1)
                                else: armor = self.get_bases_armor(self.cars[i], 0)
                            x, y = armor - self.cars[n, 1:3]
                            angle = np.angle(x+y*1j, deg=True) - self.cars[n, 4] - self.cars[n, 3]
                            if angle >= 180: angle -= 360
                            if angle <= -180: angle += 360
                            angles[ii] = angle
                        m = np.where(np.abs(angles) == np.abs(angles).min())
                        self.cars[n, 4] += angles[m][0]
                        # self.cars[n, 4] += targets[0][0]
                        if self.cars[n, 4] > 180: self.cars[n, 4] -= 360
                        if self.cars[n, 4] < -180: self.cars[n, 4] += 360
            # move x and y
            if self.acts[n, 2] or self.acts[n, 3]:
                angle = np.deg2rad(self.cars[n, 3])
                # x
                p = self.cars[n, 1]
                self.cars[n, 1] += (self.acts[n, 2]) * np.cos(angle) - (self.acts[n, 3]) * np.sin(angle)
                if self.check_interface(n):
                    self.acts[n, 2] = -self.acts[n, 2] * self.move_discount
                    self.cars[n, 1] = p
                # y
                p = self.cars[n, 2]
                self.cars[n, 2] += (self.acts[n, 2]) * np.sin(angle) + (self.acts[n, 3]) * np.cos(angle)
                if self.check_interface(n):
                    self.acts[n, 3] = -self.acts[n, 3] * self.move_discount
                    self.cars[n, 2] = p
            # fire or not
            if self.acts[n, 4] and self.cars[n, 10]:
                if self.cars[n, 9]:
                    self.cars[n, 10] -= 1
                    self.bullets.append(bullet(self.cars[n, 1:3], self.cars[n, 4]+self.cars[n, 3], self.bullet_speed, n))
                    self.cars[n, 5] += self.bullet_speed
                    self.cars[n, 9] = 0
                else:
                    self.cars[n, 9] = 1
            else:
                self.cars[n, 9] = 1
        elif self.cars[n, 7] < 0: assert False
        else:
            self.cars[n, 7] -= 1
            if self.cars[n, 7] == 0:
                self.cars[n, 8] == 0
        # check supply
        if self.acts[n, 6]:
            dis = np.abs(self.cars[n, 1:3] - [self.areas[int(self.cars[n, 0]), 1][0:2].mean(), \
                                   self.areas[int(self.cars[n, 0]), 1][2:4].mean()]).sum()
            if dis < 46 and self.compet_info[int(self.cars[n, 0]), 0] and not self.cars[n, 7]:
                self.cars[n, 8] = 1
                self.cars[n, 7] = 100 # 0.5s
                self.cars[n, 10] += 50
                self.compet_info[int(self.cars[n, 0]), 0] -= 1

#子弹和场地元素的交互逻辑，包括了和障碍物、墙体、车体、基地的交互逻辑，其中只有命中了装甲板的才会对对应单位扣除血量
    def move_bullet(self, n):
        '''
        move bullet No.n, if interface with wall, barriers or cars, return True, else False
        if interface with cars, cars'hp will decrease
        '''
        old_point = self.bullets[n].center.copy()
        self.bullets[n].center[0] += self.bullets[n].speed * np.cos(np.deg2rad(self.bullets[n].angle))
        self.bullets[n].center[1] += self.bullets[n].speed * np.sin(np.deg2rad(self.bullets[n].angle))
        # bullet wall check
        if self.bullets[n].center[0] <= 0 or self.bullets[n].center[0] >= self.map_length \
            or self.bullets[n].center[1] <= 0 or self.bullets[n].center[1] >= self.map_width: return True
        # bullet barrier check
        for b in self.barriers:
            if self.line_barriers_check(self.bullets[n].center, old_point): return True
        # bullet armor check
        for i in range(len(self.cars)):
            if i == self.bullets[n].owner: continue
            if i < self.car_num:
                if np.abs(np.array(self.bullets[n].center) - np.array(self.cars[i, 1:3])).sum() < 52.5:
                    points = self.transfer_to_car_coordinate(np.array([self.bullets[n].center, old_point]), i)
                    if self.segment(points[0], points[1], [-18.5, -5], [-18.5, 6]) \
                    or self.segment(points[0], points[1], [18.5, -5], [18.5, 6]) \
                    or self.segment(points[0], points[1], [-5, 30], [5, 30]) \
                    or self.segment(points[0], points[1], [-5, -30], [5, -30]):
                        # if self.compet_info[int(self.cars[i, 0]), 3]: self.cars[i, 6] -= 25
                        # else: self.cars[i, 6] -= 10
                        self.cars[i, 6] -= 10
                        return True
                    if self.line_rect_check(points[0], points[1], [-18, -29, 18, 29]): return True
            else: #基地的装甲板判断扣血逻辑
                if np.abs(np.array(self.bullets[n].center) - np.array(self.cars[i, 1:3])).sum() < 52.5:
                    points = self.transfer_to_car_coordinate(np.array([self.bullets[n].center, old_point]), i)
                    if self.segment(points[0], points[1], [-35, -30], [-48, 13]) \
                    or self.segment(points[0], points[1], [-12, 46], [30, 34]) \
                    or self.segment(points[0], points[1], [10, -41], [42, -9]): 
                        if self.cars[i,15] > 0 :
                            self.cars[i,15] -= 5
                        else:
                            self.cars[i, 6] -= 5
                        return True
                    
                    if self.line_triangle_check(points[0], points[1],  [-20, -70, -52, 55, 70, 22]): return True
        return False
    
    def check_hit(self):
        min_distance = float('inf')
        for n in range(self.car_num):
            target = None
            self.atk[n] = 0
            for i in range(self.car_num-1):
                if self.cars[n, 0] == self.cars[n-i-1, 0]:  # same team
                    continue
                if self.vision[n, n-i-1] == 0:  # not in vision
                    continue
                distance = np.linalg.norm(self.cars[n, 1:3] - self.cars[n-i-1, 1:3])
                if distance < min_distance:
                    min_distance = distance
                    target = n-i-1
            if target is not None:
                self.attack(n,target)
        

    def attack(self, a, b):
        distance = np.linalg.norm(self.cars[a, 1:3] - self.cars[b, 1:3])
        if a == 0:   #如果为我们的哨兵
            optimal_distance = 340  # replace with your optimal distance
            std_dev = 500  # replace with your standard deviation
            hit_probability = norm.pdf(distance, optimal_distance, std_dev) 
            max_probability = norm.pdf(optimal_distance, optimal_distance, std_dev)
            normalized_hit_probability = hit_probability / max_probability
            random_number = np.random.random()
            if random_number < normalized_hit_probability *0.5:
                self.atk[a,b] = 1
                self.cars[b,6] -= 10
                print(1)
            else:
                # missed the target
                pass  # replace with your code
            

    def update_display(self):
        assert self.render, 'only render mode need update_display'
        self.screen.fill(self.gray)
        self.screen.blit(self.background, (0, 0))
        # for i in range(len(self.barriers_rect)):
        #     self.screen.blit(self.barriers_img[i], self.barriers_rect[i])
        # for i in range(len(self.areas_rect)):
        #     self.screen.blit(self.areas_img[i], self.areas_rect[i]) 
        for i in range(len(self.bullets)):
            self.bullet_rect.center = self.bullets[i].center
            self.screen.blit(self.bullet_img, self.bullet_rect)
        for n in range(self.car_num):
            chassis_rotate = pygame.transform.rotate(self.chassis_img, -self.cars[n, 3]-90)
            gimbal_rotate = pygame.transform.rotate(self.gimbal_img, -self.cars[n, 4]-self.cars[n, 3]-90)
            chassis_rotate_rect = chassis_rotate.get_rect()
            gimbal_rotate_rect = gimbal_rotate.get_rect()
            chassis_rotate_rect.center = self.cars[n, 1:3]
            gimbal_rotate_rect.center = self.cars[n, 1:3]
            self.screen.blit(chassis_rotate, chassis_rotate_rect)
            self.screen.blit(gimbal_rotate, gimbal_rotate_rect)
        # self.screen.blit(self.base_blue_img, self.base_blue_rect)
        # self.screen.blit(self.base_red_img, self.base_red_rect)
        for n in range(self.car_num):
            if n < self.car_num:
                select = np.where((self.vision[n] == 1))[0]+1
                select2 = np.where((self.detect[n] == 1))[0]+1
                info = self.font.render('{} | {}: {} {}'.format(int(self.cars[n, 6]), n+1, select, select2), True, self.blue if self.cars[n, 0] else self.red)
                self.screen.blit(info, self.cars[n, 1:3]+[-20, -60])
                info = self.font.render('{} {}'.format(int(self.cars[n, 10]), int(self.cars[n, 5])), True, self.blue if self.cars[n, 0] else self.red)
                self.screen.blit(info, self.cars[n, 1:3]+[-20, -45])
            else:
                info = self.font.render('s{} h{}'.format(int(self.cars[n, 15]),int(self.cars[n, 6])), True, self.blue if self.cars[n, 0] else self.red)
                self.screen.blit(info, self.cars[n, 1:3]+[-25, -65])
        for n in range(self.car_num):
            for m in range(self.car_num):
                if m != n and self.atk[n,m] == 1:  # if another car is visible
                    start = [round(x) for x in self.cars[n, 1:3]]
                    end = [round(x) for x in self.cars[m, 1:3]]
                    color = self.blue if self.cars[n, 0] else self.red
                    pygame.draw.line(self.screen, color, start, end, 1)  # draw a line between the two cars
        info = self.font.render('time: {}'.format(self.time), False, (0, 0, 0))
        self.screen.blit(info, (8, 8))
        if self.dev: self.dev_window()
        pygame.display.flip()

    def dev_window(self):
        for n in range(self.car_num):
            wheels = self.check_points_wheel(self.cars[n])
            for w in wheels:
                pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, w.astype(int), 3)
            armors = self.check_points_armor(self.cars[n])
            for a in armors:
                pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, a.astype(int), 3)
        self.screen.blit(self.info_bar_img, self.info_bar_rect)
        for n in range(self.car_num):
            tags = ['owner', 'x', 'y', 'angle', 'yaw', 'heat', 'hp', 'freeze_time', 'is_supply', 
                    'can_shoot', 'bullet', 'stay_time', 'wheel_hit', 'armor_hit', 'car_hit', 'sheild', 'invincible', 'pre_hp']
            info = self.font.render('car {}'.format(n), False, (0, 0, 0))
            self.screen.blit(info, (8+n*100, 100))
            for i in range(self.cars[n].size-2):
                info = self.font.render('{}: {}'.format(tags[i], int(self.cars[n, i])), False, (0, 0, 0))
                self.screen.blit(info, (8+n*100, 117+i*17))
        info = self.font.render('red   supply: {}   bonus: {}   bonus_time: {}   progress:{}'.format(self.compet_info[0, 0], \
                                self.compet_info[0, 1], self.compet_info[0, 2], self.compet_info[0, 3]), False, (0, 0, 0))
        self.screen.blit(info, (18, 400))#372
        info = self.font.render('blue   supply: {}   bonus: {}   bonus_time: {}   progress:{}'.format(self.compet_info[1, 0], \
                                self.compet_info[1, 1], self.compet_info[1, 2], self.compet_info[1, 3]), False, (0, 0, 0))
        self.screen.blit(info, (18, 389))

    def get_order(self): 
        # get order from controler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_1]: self.n = 0
        if pressed[pygame.K_2]: self.n = 1
        if pressed[pygame.K_3]: self.n = 2
        if pressed[pygame.K_4]: self.n = 3
        self.orders[self.n] = 0
        if pressed[pygame.K_w]: self.orders[self.n, 0] += 1
        if pressed[pygame.K_s]: self.orders[self.n, 0] -= 1
        if pressed[pygame.K_q]: self.orders[self.n, 1] -= 1
        if pressed[pygame.K_e]: self.orders[self.n, 1] += 1
        if pressed[pygame.K_a]: self.orders[self.n, 2] -= 1
        if pressed[pygame.K_d]: self.orders[self.n, 2] += 1
        if pressed[pygame.K_b]: self.orders[self.n, 3] -= 1
        if pressed[pygame.K_m]: self.orders[self.n, 3] += 1
        if pressed[pygame.K_SPACE]: self.orders[self.n, 4] = 1
        else: self.orders[self.n, 4] = 0
        if pressed[pygame.K_f]: self.orders[self.n, 5] = 1
        else: self.orders[self.n, 5] = 0
        if pressed[pygame.K_r]: self.orders[self.n, 6] = 1
        else: self.orders[self.n, 6] = 0
        if pressed[pygame.K_n]: self.orders[self.n, 7] = 1
        else: self.orders[self.n, 7] = 0
        if pressed[pygame.K_TAB]: self.dev = True
        else: self.dev = False
        return False

    def orders_to_acts(self, n):
        # turn orders to acts
        # print(self.acts.shape)
        self.acts[n,2] += self.orders[n,0] * 1.5 / self.motion
        if self.orders[n, 0] == 0:
            if self.acts[n, 2] > 0: self.acts[n, 2] -= 1.5 / self.motion
            if self.acts[n, 2] < 0: self.acts[n, 2] += 1.5 / self.motion
        if abs(self.acts[n, 2]) < 1.5 / self.motion: self.acts[n, 2] = 0
        if self.acts[n, 2] >= 1.5: self.acts[n, 2] = 1.5
        if self.acts[n, 2] <= -1.5: self.acts[n, 2] = -1.5
        # x, y
        self.acts[n, 3] += self.orders[n, 1] * 1 / self.motion
        if self.orders[n, 1] == 0:
            if self.acts[n, 3] > 0: self.acts[n, 3] -= 1 / self.motion
            if self.acts[n, 3] < 0: self.acts[n, 3] += 1 / self.motion
        if abs(self.acts[n, 3]) < 1 / self.motion: self.acts[n, 3] = 0
        if self.acts[n, 3] >= 1: self.acts[n, 3] = 1
        if self.acts[n, 3] <= -1: self.acts[n, 3] = -1
        # rotate chassis
        self.acts[n, 0] += self.orders[n, 2] * 1 / self.rotate_motion
        if self.orders[n, 2] == 0:
            if self.acts[n, 0] > 0: self.acts[n, 0] -= 1 / self.rotate_motion
            if self.acts[n, 0] < 0: self.acts[n, 0] += 1 / self.rotate_motion
        if abs(self.acts[n, 0]) < 1 / self.rotate_motion: self.acts[n, 0] = 0
        if self.acts[n, 0] > 1: self.acts[n, 0] = 1
        if self.acts[n, 0] < -1: self.acts[n, 0] = -1
        # rotate yaw
        self.acts[n, 1] += self.orders[n, 3] / self.yaw_motion
        if self.orders[n, 3] == 0:
            if self.acts[n, 1] > 0: self.acts[n, 1] -= 1 / self.yaw_motion
            if self.acts[n, 1] < 0: self.acts[n, 1] += 1 / self.yaw_motion
        if abs(self.acts[n, 1]) < 1 / self.yaw_motion: self.acts[n, 1] = 0
        if self.acts[n, 1] > 3: self.acts[n, 1] = 3
        if self.acts[n, 1] < -3: self.acts[n, 1] = -3
        self.acts[n, 4] = self.orders[n, 4]
        self.acts[n, 6] = self.orders[n, 5]
        self.acts[n, 5] = self.orders[n, 6]
        self.acts[n, 7] = self.orders[n, 7]

    def set_car_loc(self, n, loc):
        self.cars[n, 1:3] = loc

    def get_map(self):
        return g_map(self.map_length, self.map_width, self.areas, self.barriers)

    def stay_check(self):
        # check bonus stay
        for n in range(self.cars.shape[0]):
            a = self.areas[int(self.cars[n, 0]), 0]
            if self.compet_info[int(self.cars[n, 0]), 2] > 0:
                    self.compet_info[int(self.cars[n, 0]), 2] -= 1
            if self.cars[n, 1] >= a[0] and self.cars[n, 1] <= a[1] and self.cars[n, 2] >= a[2] \
            and self.cars[n, 2] <= a[3] and self.compet_info[int(self.cars[n, 0]), 1] \
            and self.compet_info[int(self.cars[n, 0]), 2] == 0:
                self.cars[n, 11] += 1 # 1/200 s
                # if self.cars[n, 11] >= 1000: # 5s
                #     self.cars[n, 11] = 0
                #     self.compet_info[int(self.cars[n, 0]), 2] = 18000 # 90s
                
                if self.cars[n, 11] % 200 == 0: # 每1s加10进度
                    self.compet_info[int(self.cars[n, 0]), 3] += 10
                    if self.compet_info[int(self.cars[n, 0]), 3] >= 100:
                        self.compet_info[0, 2] = 18000 # 增益点双方共同失效90s
                        self.compet_info[1, 2] = 18000 
                        self.compet_info[0, 3] = 0 # 双方占领进度清零
                        self.compet_info[1, 3] = 0
                
                if self.cars[n, 6] < self.cars[n, 12]:
                    if self.compet_info[int(self.cars[n, 0]), 3] - 2 < 0:
                        self.compet_info[int(self.cars[n, 0]), 3] = 0
                    else:
                        self.compet_info[int(self.cars[n, 0]), 3] -= 2 * ((self.cars[n, 12] - self.cars[n, 6]) // 10)# 每少10点血量扣两点进度
                self.cars[n, 12] = self.cars[n, 6]#更新上一时刻的血量
            else: 
                self.cars[n, 11] = 0
                self.cars[n, 12] = self.cars[n, 6]
        for i in range(2):
            if self.compet_info[i, 2] > 0:
                self.compet_info[i, 2] -= 1

    def cross(self, p1, p2, p3):          #向量p1p2 叉乘 向量p1p3 如果结果为正，那么p1p2在p1p3的顺时针方向；如果结果为负，那么p1p2在p1p3的逆时针方向；如果结果为0，那么p1p2和p1p3共线。
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1 

    def segment(self, p1, p2, p3, p4):     #判断p1p2与p3p4是否相交
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        if (max(p1[0], p2[0])>=min(p3[0], p4[0]) and max(p3[0], p4[0])>=min(p1[0], p2[0])
        and max(p1[1], p2[1])>=min(p3[1], p4[1]) and max(p3[1], p4[1])>=min(p1[1], p2[1])):
            if (self.cross(p1,p2,p3)*self.cross(p1,p2,p4)<=0 and self.cross(p3,p4,p1)*self.cross(p3,p4,p2)<=0): return True
            else: return False
        else: return False

    def line_region_check(self,l1,l2,rg):  #检查线与区域是否相交
        for i in range(len(rg)):
            # 如果是最后一个元素，那么必然是和第一个一样的，就啥也不干
            if i+1 == len(rg):
                break
            if self.segment(l1,l2,rg[i],rg[i + 1]) : 
                return True
            else: return False

    def line_rect_check(self, l1, l2, sq):   #检查线与方形区域是否相交
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        # check if line cross rect, sq = [x_leftdown, y_leftdown, x_rightup, y_rightup]
        p1 = [sq[0], sq[1]]  
        p2 = [sq[2], sq[3]]
        p3 = [sq[2], sq[1]]
        p4 = [sq[0], sq[3]]
        if self.segment(l1,l2,p1,p2) or self.segment(l1,l2,p3,p4): return True
        else: return False

    def line_triangle_check(self, l1, l2, tri):
        # check if line cross triangle, tri = [x1, y1, x2, y2, x3, y3]
        p1 = [tri[0], tri[1]]
        p2 = [tri[2], tri[3]]
        p3 = [tri[4], tri[5]]
        if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p2, p3) or self.segment(l1, l2, p3, p1): 
            return True
        else: 
            return False

    def line_barriers_check(self, l1, l2):
        for rg in self.region:
            if self.line_region_check(l1, l2, rg): 
                return True
                
        return False

    def line_cars_check(self, l1, l2):
        for car in self.cars:
            if (car[1:3] == l1).all() or (car[1:3] == l2).all():
                continue
            p1, p2, p3, p4 = self.get_car_outline(car)
            if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p3, p4): return True
        return False

    def get_lidar_vision(self):
        for n in range(self.car_num):
            for i in range(self.car_num-1):
                x, y = self.cars[n-i-1, 1:3] - self.cars[n, 1:3]
                angle = np.angle(x+y*1j, deg=True)
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                angle = angle - self.cars[n, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                if abs(angle) < self.lidar_angle:
                    if self.line_barriers_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]) \
                    or self.line_cars_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]):
                        self.detect[n, n-i-1] = 0
                    else: self.detect[n, n-i-1] = 1
                else: self.detect[n, n-i-1] = 0

    def get_camera_vision(self):                                         #视野
        for n in range(self.car_num):
            for i in range(self.car_num-1):
                x, y = self.cars[n-i-1, 1:3] - self.cars[n, 1:3]
                angle = np.angle(x+y*1j, deg=True)
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                angle = angle - self.cars[n, 4] - self.cars[n, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                if abs(angle) < self.camera_angle:
                    if self.line_barriers_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]) \
                    or self.line_cars_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]):
                        self.vision[n, n-i-1] = 0
                    else: self.vision[n, n-i-1] = 1
                else: self.vision[n, n-i-1] = 0            

    def transfer_to_car_coordinate(self, points, n):
        
        pan_vecter = -self.cars[n, 1:3]
        rotate_matrix = np.array([[np.cos(np.deg2rad(self.cars[n, 3]+90)), -np.sin(np.deg2rad(self.cars[n, 3]+90))],
                                [np.sin(np.deg2rad(self.cars[n, 3]+90)), np.cos(np.deg2rad(self.cars[n, 3]+90))]])
        return np.matmul(points + pan_vecter, rotate_matrix)
        


    def check_points_wheel(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-22.5, -29], [22.5, -29], 
                       [-22.5, -14], [22.5, -14], 
                       [-22.5, 14], [22.5, 14],
                       [-22.5, 29], [22.5, 29]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]

    def check_points_armor(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-6.5, -30], [6.5, -30], 
             [-18.5,  -7], [18.5,  -7],
             [-18.5,  0], [18.5,  0],
             [-18.5,  6], [18.5,  6],
             [-6.5, 30], [6.5, 30]])
        return [np.matmul(x, rotate_matrix) + car[1:3] for x in xs]
    
    def check_points_base_armor(self, base):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(base[3]+90)), -np.sin(-np.deg2rad(base[3]+90))],
                                  [np.sin(-np.deg2rad(base[3]+90)), np.cos(-np.deg2rad(base[3]+90))]])
        xs = np.array([[-18, -18], [5, -25], 
             [-24,  5], [-5,  23],
             [17,  16], [25,  -5]])
        return [np.matmul(x, rotate_matrix) + base[1:3] for x in xs]

    def get_car_outline(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-22.5, -30], [22.5, 30], [-22.5, 30], [22.5, -30]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]
    

    def isInRegion(self,lon, lat, region_1):
        '''
        判断点[lon, lat] 是否在区域 region_1内
        :param lon: 输入点的 经度
        :param lat: 输入点的 纬度
        :param region_1: region1是一个list，其格式为：[[-74.035, 40.695],[-74.036, 40.694]...] 其内层list中第一个元素代表经度，第二个代表纬度
        :return: True则在所选区域内  False则不在
        '''
        count = 0
        lat = float(lat)
        lon = float(lon)

        for i in range(len(region_1)):
            # 如果是最后一个元素，那么必然是和第一个一样的，就啥也不干
            if i+1 == len(region_1):
                break
            # 如果不是最后一个元素，那么需要和后一个元素一起判断给定点是否在区域内
            lo_1, la_1 = region_1[i]
            lo_2, la_2 = region_1[i + 1]

            la_1, lo_1, la_2, lo_2 = float(la_1), float(lo_1), float(la_2), float(lo_2)
            # print((lo_1, la_1))
            # # 以纬度确定位置，沿纬度向右作射线，看交点个数
            if lat < min(la_1, la_2):
                # print('点高度小于线段', (lo_1, la_1))
                continue
            if lat > max(la_1, la_2):
                # print('点高度大于线段', (lo_1, la_1))
                continue
            # 如果和某一个共点那么直接返回true
            if (lat, lon) == (la_1, lo_1) or (lat, lon) == (la_2, lo_2):
                # print('在线段顶点上', (lo_1, la_1))
                return True
            # 如果和两点共线
            if lat == la_1 == la_2:
                # print('两点共线', (lo_1, la_1))
                continue

            # 接下来只需考虑射线穿越的情况，该情况下的特殊情况是射线穿越顶点
            # 求交点的经度
            cross_lon = (lat - la_1) * (lo_2 - lo_1) / (la_2 - la_1 or 0.000000000000000000000001) + lo_1
            # 无所谓在交点在点的左右 方向向上的边不包括其终止点  方向向下的边不包括其开始点
            if lat == max(la_1, la_2):
                continue
            # 其他情况
            elif cross_lon > lon:
                count += 1

        # print(count)
        if count%2 == 0:
            return False
        return True



    def check_interface(self, n):
        # car barriers assess
        wheels = self.check_points_wheel(self.cars[n])
        for w in wheels:
            if w[0] <= 0 or w[0] >= self.map_length or w[1] <= 0 or w[1] >= self.map_width:
                self.cars[n, 12] += 1
                return True
            # for b in self.barriers:
            #     if w[0] >= b[0] and w[0] <= b[1] and w[1] >= b[2] and w[1] <= b[3]:
            #         self.cars[n, 12] += 1
            #         return True
            for r in self.region:
                if self.isInRegion(w[0], w[1], r):
                    self.cars[n, 12] += 1
                    return True
        armors = self.check_points_armor(self.cars[n])
        for a in armors:
            if a[0] <= 0 or a[0] >= self.map_length or a[1] <= 0 or a[1] >= self.map_width:
                self.cars[n, 13] += 1
                self.cars[n, 6] -= 10
                return True
            # for b in self.barriers:
            #     if a[0] >= b[0] and a[0] <= b[1] and a[1] >= b[2] and a[1] <= b[3]:
            #         self.cars[n, 13] += 1
            #         self.cars[n, 6] -= 10
            #         return True
                
        
        # car car assess 待去除基地装甲板
        for i in range(self.car_num):
            if i == n: continue
            wheels_tran = self.transfer_to_car_coordinate(wheels, i)
            for w in wheels_tran:
                if w[0] >= -22.5 and w[0] <= 22.5 and w[1] >= -30 and w[1] <= 30:
                    self.cars[n, 14] += 1
                    return True
            armors_tran = self.transfer_to_car_coordinate(armors, i)
            for a in armors_tran:
                if a[0] >= -22.5 and a[0] <= 22.5 and a[1] >= -30 and a[1] <= 30:
                    self.cars[n, 14] += 1
                    self.cars[n, 6] -= 10
                    return True
        return False

    def get_armor(self, car, i):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[0, -30], [18.5, 0], [0, 30], [-18.5,  0]])
        # [-6.5, -30], [6.5, -30], 
        #      [-18.5,  -7], [18.5,  -7],
        #      [-18.5,  0], [18.5,  0],
        #      [-18.5,  6], [18.5,  6],
        #      [-6.5, 30], [6.5, 30]
        return np.matmul(xs[i], rotate_matrix) + car[1:3]
    
    def get_bases_armor(self, base, i):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(base[3]+90)), -np.sin(-np.deg2rad(base[3]+90))],
                                  [np.sin(-np.deg2rad(base[3]+90)), np.cos(-np.deg2rad(base[3]+90))]])
        xs = np.array([[0, -30], [18.5, 0], [0, 30], [-18.5,  0]])
        return np.matmul(xs[i], rotate_matrix) + base[1:3]

    def save_record(self, file):
        np.save(file, self.memory)
            
            
''' important indexs
areas_index = [[{'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 0 bonus red
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 1 supply red
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 2 start 0 red
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}], # 3 start 1 red

               [{'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 0 bonus blue
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 1 supply blue
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 2 start 0 blue
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}]] # 3 start 1 blue
                            

barriers_index = [{'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 0 horizontal
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 1 horizontal
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 2 horizontal
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 3 vertical
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 4 vertical
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 5 vertical
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}] # 6 vertical

# armor编号：0：前，1：右，2：后，3左，车头为前

act_index = {'rotate_speed': 0, 'yaw_speed': 1, 'x_speed': 2, 'y_speed': 3, 'shoot': 4, 'shoot_mutiple': 5, 'supply': 6,
             'auto_aim': 7}

bullet_speed: 12.5


compet_info_index = {'red': {'supply': 0, 'bonus': 1, 'bonus_stay_time(deprecated)': 2, 'bonus_time': 3}, 
                     'blue': {'supply': 0, 'bonus': 1, 'bonus_stay_time(deprecated)': 2, 'bonus_time': 3}}
int, shape: (2, 4)

order_index = ['x', 'y', 'rotate', 'yaw', 'shoot', 'supply', 'shoot_mode', 'auto_aim']
int, shape: (8,)
    x, -1: back, 0: no, 1: head
    y, -1: left, 0: no, 1: right
    rotate, -1: anti-clockwise, 0: no, 1: clockwise, for chassis
    shoot_mode, 0: single, 1: mutiple
    shoot, 0: not shoot, 1: shoot
    yaw, -1: anti-clockwise, 0: no, 1: clockwise, for gimbal
    auto_aim, 0: not, 1: auto aim

car_index = {"owner": 0, 'x': 1, 'y': 2, "angle": 3, "yaw": 4, "heat": 5, "hp": 6, 
             "freeze_time": 7, "is_supply": 8, "can_shoot": 9, 'bullet': 10, 'stay_time': 11,
             'wheel_hit': 12, 'armor_hit': 13, 'car_hit': 14}
float, shape: (14,)

'''

    
#增益区机制修改，增益区占领进度增加速度加快，每1s增加10进度，每少10点血量扣两点进度