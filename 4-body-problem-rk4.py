import pyxel
import math

G = 1       # 重力定数
DT = 0.001   # 時間のステップ幅
MAX_TRAIL = 100  # 各天体の軌道記録数（点の最大数）
STEPS = 1_000_000  # シミュレーションのステップ数
SKIP_FRAMES = 1000     # 表示をさらに早送りするためのスキップフレーム数を短縮

class Body:
    def __init__(self, x, y, vx, vy, mass, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.color = color
        self.trail = []  # 軌跡用リスト

    def compute_acceleration(self, bodies):
        ax, ay = 0, 0
        for other in bodies:
            if self == other:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist_sq = dx * dx + dy * dy
            dist = math.sqrt(dist_sq) + 1e-4  # 0除算防止
            force = G * self.mass * other.mass / dist_sq
            ax += force * dx / dist / self.mass
            ay += force * dy / dist / self.mass
        return ax, ay

    def update_position_rk4(self, bodies):
        def rk4_step(x, v, a, dt):
            k1x = v
            k1v = a
            k2x = v + 0.5 * dt * k1v
            k2v = a  # 加速度は一定と仮定
            k3x = v + 0.5 * dt * k2v
            k3v = a
            k4x = v + dt * k3v
            k4v = a
            x_next = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
            v_next = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
            return x_next, v_next

        ax, ay = self.compute_acceleration(bodies)
        self.x, self.vx = rk4_step(self.x, self.vx, ax, DT)
        self.y, self.vy = rk4_step(self.y, self.vy, ay, DT)
        self.trail.append((self.x, self.y))
        # 軌跡の長さを制限
        if len(self.trail) > MAX_TRAIL:
            self.trail.pop(0)

class App:
    def __init__(self):
        pyxel.init(200, 200, title="Three-Body Simulation")
        
        # 初期状態の設定
        center_mass = 50
        orbit_radius = 40
        orbit_speed = math.sqrt(G * center_mass / orbit_radius)
        small_mass = center_mass / 100_000  # 4つ目の天体の質量

        # 4つ目の天体を赤色の中心天体の衛星軌道に配置
        satellite_orbit_radius = 10
        satellite_orbit_speed = math.sqrt(G * center_mass / satellite_orbit_radius)

        self.bodies = [
            Body(100, 100, 0, 0, center_mass, 8),   # 赤: 中心で静止
            Body(100 - orbit_radius, 100, 0, orbit_speed, 10, 9),  # 緑: 円軌道
            Body(100 + orbit_radius, 100, 0, -orbit_speed, 10, 10), # 青: 円軌道
            Body(100 + satellite_orbit_radius, 100, 0, -satellite_orbit_speed, small_mass, 11)  # 黄: 衛星軌道
        ]

        # シミュレーション結果を事前に計算
        self.simulation_data = self.precompute_simulation()
        self.current_frame = 0

        pyxel.run(self.update, self.draw)

    def precompute_simulation(self):
        simulation_data = []
        bodies_copy = [
            Body(body.x, body.y, body.vx, body.vy, body.mass, body.color)
            for body in self.bodies
        ]
        for step in range(STEPS):
            step_data = [(body.x, body.y) for body in bodies_copy]
            simulation_data.append(step_data)
            for body in bodies_copy:
                body.update_position_rk4(bodies_copy)
            
            # 実行状況を端末に出力
            if step % (STEPS // 10) == 0:  # 進捗を10%ごとに表示
                print(f"Precomputing simulation: {step * 100 // STEPS}% completed")

        print("Precomputing simulation: 100.0% completed")
        return simulation_data

    def update(self):
        # フレームを進める
        self.current_frame += SKIP_FRAMES  # スキップフレームを適用
        if self.current_frame >= len(self.simulation_data):
            self.current_frame = 0  # ループ再生

        # 軌跡をスキップフレームに対応して更新
        for body in self.bodies:
            body.trail = body.trail[-MAX_TRAIL:]  # 軌跡の長さを制限

    def draw(self):
        pyxel.cls(0)
        frame_data = self.simulation_data[self.current_frame]
        for i, body in enumerate(self.bodies):
            body.trail.append(frame_data[i])
            body.trail = body.trail[-MAX_TRAIL:]  # 軌跡の長さを制限
            for x, y in body.trail:
                pyxel.pset(int(x), int(y), body.color)
            # 4つ目の天体は小さく描画
            size = 2 if i < 3 else 1
            pyxel.circ(int(frame_data[i][0]), int(frame_data[i][1]), size, body.color)
        
        # 現在のフレーム数を右下に表示
        frame_text = f"{self.current_frame}/{STEPS}"
        pyxel.text(pyxel.width - len(frame_text) * 4 - 2, pyxel.height - 8, frame_text, 7)

App()
