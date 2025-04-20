import pyxel
import math

G = 1       # 重力定数
DT = 0.01   # 初期時間のステップ幅
TOL = 5e-7  # 許容誤差
MAX_TRAIL = 100  # 各天体の軌道記録数（点の最大数）
STEPS = 20_000  # シミュレーションのステップ数
SKIP_FRAMES = 30     # 表示をさらに早送りするためのスキップフレーム数を短縮

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

    def update_position_adaptive_rk(self, bodies):
        def rkf45_step(x, v, a_func, dt):
            # 4次と5次のRunge-Kutta計算
            k1x, k1v = v, a_func(x, v)
            k2x, k2v = v + 0.25 * dt * k1v, a_func(x + 0.25 * dt * k1x, v + 0.25 * dt * k1v)
            k3x, k3v = v + (3/8) * dt * k2v, a_func(x + (3/8) * dt * k2x, v + (3/8) * dt * k2v)
            k4x, k4v = v + (12/13) * dt * k3v, a_func(x + (12/13) * dt * k3x, v + (12/13) * dt * k3v)
            k5x, k5v = v + dt * k4v, a_func(x + dt * k4x, v + dt * k4v)
            k6x, k6v = v + 0.5 * dt * k5v, a_func(x + 0.5 * dt * k5x, v + 0.5 * dt * k5v)

            x4 = x + dt * (25/216 * k1x + 1408/2565 * k3x + 2197/4104 * k4x - 1/5 * k5x)
            v4 = v + dt * (25/216 * k1v + 1408/2565 * k3v + 2197/4104 * k4v - 1/5 * k5v)

            x5 = x + dt * (16/135 * k1x + 6656/12825 * k3x + 28561/56430 * k4x - 9/50 * k5x + 2/55 * k6x)
            v5 = v + dt * (16/135 * k1v + 6656/12825 * k3v + 28561/56430 * k4v - 9/50 * k5v + 2/55 * k6v)

            error = max(abs(x5 - x4), abs(v5 - v4))
            return x5, v5, error

        def acceleration(x, y, v):
            ax, ay = 0, 0
            for other in bodies:
                if self == other:
                    continue
                dx = other.x - x
                dy = other.y - y
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq) + 1e-4  # 0除算防止
                force = G * self.mass * other.mass / dist_sq
                ax += force * dx / dist / self.mass
                ay += force * dy / dist / self.mass
            return ax, ay

        global DT
        while True:
            x_next, vx_next, error_x = rkf45_step(self.x, self.vx, lambda x, v: acceleration(x, self.y, v)[0], DT)
            y_next, vy_next, error_y = rkf45_step(self.y, self.vy, lambda y, v: acceleration(self.x, y, v)[1], DT)
            error = max(error_x, error_y)

            if error <= TOL:
                self.x, self.vx = x_next, vx_next
                self.y, self.vy = y_next, vy_next
                self.trail.append((self.x, self.y))
                if len(self.trail) > MAX_TRAIL:
                    self.trail.pop(0)
                # エラーがゼロの場合にステップ幅を増加
                if error == 0:
                    DT *= 2
                else:
                    DT *= min(2, max(0.5, (TOL / error) ** 0.2))  # 次のステップ幅を調整
                break
            else:
                DT *= 0.5  # ステップ幅を縮小

class App:
    def __init__(self):
        pyxel.init(500, 500, title="Four-Body Simulation")  # タイトルを変更
        
        # 初期状態の設定
        center_mass = 50
        orbit_radius = 40
        orbit_speed = math.sqrt(G * center_mass / orbit_radius)
        small_mass = center_mass / 100_000  # 4つ目の天体の質量

        # 4つ目の天体を赤色の中心天体の衛星軌道に配置
        satellite_orbit_radius = 20
        satellite_orbit_speed = math.sqrt(G * center_mass / satellite_orbit_radius)

        self.bodies = [
            Body(200, 200, 0, 0, center_mass, 8),   # 赤: 中心で静止
            Body(200 - orbit_radius, 200, 0, orbit_speed, 10, 9),  # 緑: 円軌道
            Body(200 + orbit_radius, 200, 0, -orbit_speed, 10, 10), # 青: 円軌道
            Body(200 + satellite_orbit_radius, 200, 0, -satellite_orbit_speed, small_mass, 11)  # 黄: 衛星軌道
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
                body.update_position_adaptive_rk(bodies_copy)
            
            # 実行状況を端末に出力（進捗を20%ごとに表示）
            if step % (STEPS // 5) == 0:  # 進捗を20%ごとに表示
                print(f"Precomputing simulation: {step / STEPS * 100:.0f}% completed")
            
            # デバッグ用ログ（必要に応じて有効化）
            # print(f"Step {step}: Body positions: {[ (b.x, b.y) for b in bodies_copy ]}")

        print("Precomputing simulation: 100% completed")
        return simulation_data

    def update(self):
        # フレームを進める
        self.current_frame += SKIP_FRAMES  # スキップフレームを適用
        if self.current_frame >= len(self.simulation_data):
            self.current_frame = 0  # ループ再生

        # 軌跡をスキップフレームに対応して更新
        for i, body in enumerate(self.bodies):
            if i < 3:  # Exclude the 4th body from trail updates
                body.trail = body.trail[-MAX_TRAIL:]  # 軌跡の長さを制限

    def draw(self):
        pyxel.cls(0)
        frame_data = self.simulation_data[self.current_frame]
        ref_body = self.bodies[3]  # 4th body as reference
        ref_x, ref_y = frame_data[3]  # Reference body's position

        for i, body in enumerate(self.bodies):
            if i < 3:  # Exclude the 4th body from trail drawing
                # Update trail with relative coordinates
                body.trail.append((frame_data[i][0] - ref_x, frame_data[i][1] - ref_y))
                body.trail = body.trail[-MAX_TRAIL:]  # 軌跡の長さを制限
                for x, y in body.trail:
                    pyxel.pset(int(x + 200), int(y + 200), body.color)  # Adjust for screen center

                pyxel.circ(int(frame_data[i][0] - ref_x + 200), int(frame_data[i][1] - ref_y + 200), 2, body.color)

        # Draw the 4th body at the center
        pyxel.circ(200, 200, 1, self.bodies[3].color)

        # 現在のフレーム数を右下に表示
        frame_text = f"{self.current_frame}/{STEPS}"
        pyxel.text(pyxel.width - len(frame_text) * 4 - 2, pyxel.height - 8, frame_text, 7)

App()
