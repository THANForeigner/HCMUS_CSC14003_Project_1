from manim import *
import numpy as np


# --- CẤU HÌNH HÀM MỤC TIÊU ---
def rastrigin(x, y):
    A = 10
    # Hàm Rastrigin: f(x) = 10n + sum(x^2 - 10cos(2pi*x))
    val = (
        A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    )
    return val


class TLBOSections(ThreeDScene):
    def construct(self):
        # ==========================================
        # SECTION 1: KHỞI TẠO MÔI TRƯỜNG
        # ==========================================
        self.next_section("Khoi tao Moi truong", skip_animations=False)

        # 1. Cấu hình Camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)

        # 2. Tạo hệ trục
        axes = ThreeDAxes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            z_range=[0, 40, 10],
            x_length=6,
            y_length=6,
            z_length=4,
        )

        # 3. Vẽ bề mặt (Surface)
        surface = Surface(
            lambda u, v: axes.c2p(u, v, rastrigin(u, v) / 8),
            u_range=[-2.1, 2.1],
            v_range=[-2.1, 2.1],
            resolution=(30, 30),
            should_make_jagged=False,
            fill_opacity=0.3,
            stroke_width=0.8,
            stroke_color=WHITE,
        )

        # Tô màu chuyển sắc (Gradient) cho bề mặt
        surface.set_fill_by_value(
            axes=axes, colors=[(BLUE, 0), (GREEN, 15), (YELLOW, 30), (RED, 60)], axis=2
        )

        self.add(axes, surface)

        # Hiệu ứng xoay nhẹ để biết là 3D
        self.begin_ambient_camera_rotation(rate=0.05)
        self.wait(1)

        # ==========================================
        # SECTION 2: KHỞI TẠO QUẦN THỂ
        # ==========================================
        self.next_section("Khoi tao Quan the", skip_animations=False)

        pop_size = 15  # Số lượng học sinh
        bounds = [-2, 2]

        # Tạo dữ liệu ngẫu nhiên
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
        dots = VGroup()

        # Hàm helper để lấy tọa độ 3D từ (x, y)
        def get_3d_point(x, y):
            z = rastrigin(x, y)
            return axes.c2p(x, y, z / 8)

        # Vẽ các chấm ban đầu
        for i in range(pop_size):
            pos = get_3d_point(population[i, 0], population[i, 1])
            dot = Dot3D(point=pos, color=RED, radius=0.08)
            dots.add(dot)

        self.play(FadeIn(dots))
        self.wait(0.5)

        # ==========================================
        # SECTION 3: VÒNG LẶP THUẬT TOÁN (150 Iterations)
        # ==========================================
        iterations = 150  # Đã tăng lên 150 vòng lặp

        for it in range(iterations):
            self.next_section(f"Iteration {it + 1}", skip_animations=False)

            # --- CƠ CHẾ TỐC ĐỘ THÍCH ỨNG (ADAPTIVE SPEED) ---
            if it < 2:
                # 2 vòng đầu: Chạy chậm rãi để xem chi tiết
                t_highlight = 1.0
                t_move_teacher = 2.0
                t_reset = 0.5
                t_move_learner = 1.5
            elif it < 15:
                # Vòng 3 đến 15: Bắt đầu tăng tốc
                t_highlight = 0.2
                t_move_teacher = 0.5
                t_reset = 0.1
                t_move_learner = 0.5
            else:
                # Từ vòng 16 trở đi: Tốc độ tối đa để nhanh chóng thấy kết quả hội tụ
                t_highlight = 0.05
                t_move_teacher = 0.1
                t_reset = 0.05
                t_move_learner = 0.1

            # Hiển thị Text số lần lặp
            iter_text = Text(f"Iteration: {it + 1}", font_size=24).to_corner(UL)
            self.add_fixed_in_frame_mobjects(iter_text)

            # --- A. TEACHER PHASE ---
            fitness = np.array([rastrigin(p[0], p[1]) for p in population])
            best_idx = np.argmin(fitness)
            teacher = population[best_idx]
            mean_pop = np.mean(population, axis=0)

            # Highlight Teacher (Màu Vàng)
            self.play(
                dots[best_idx].animate.set_color(YELLOW).scale(1.5),
                run_time=t_highlight,
            )

            new_population = population.copy()
            for i in range(pop_size):
                r = np.random.rand()
                tf = np.random.randint(1, 3)

                # Di chuyển về phía Teacher
                diff = r * (teacher - tf * mean_pop)
                new_pos = population[i] + diff
                new_pos = np.clip(new_pos, bounds[0], bounds[1])

                if rastrigin(new_pos[0], new_pos[1]) < fitness[i]:
                    new_population[i] = new_pos

            # Animation di chuyển Teacher Phase
            anims = []
            for i in range(pop_size):
                end_pos = get_3d_point(new_population[i, 0], new_population[i, 1])
                anims.append(dots[i].animate.move_to(end_pos))

            self.play(*anims, run_time=t_move_teacher)
            population = new_population

            # Reset màu Teacher về đỏ thường
            self.play(
                dots[best_idx].animate.set_color(RED).scale(1 / 1.5), run_time=t_reset
            )

            # --- B. LEARNER PHASE ---
            learner_pop = population.copy()
            current_fitness = np.array([rastrigin(p[0], p[1]) for p in population])

            for i in range(pop_size):
                partner_idx = np.random.randint(0, pop_size)
                while partner_idx == i:
                    partner_idx = np.random.randint(0, pop_size)

                Xi, Xj = population[i], population[partner_idx]

                if current_fitness[i] < current_fitness[partner_idx]:
                    diff = np.random.rand() * (Xi - Xj)
                else:
                    diff = np.random.rand() * (Xj - Xi)

                new_pos = Xi + diff
                new_pos = np.clip(new_pos, bounds[0], bounds[1])

                if rastrigin(new_pos[0], new_pos[1]) < current_fitness[i]:
                    learner_pop[i] = new_pos

            # Animation di chuyển Learner Phase
            anims = []
            for i in range(pop_size):
                end_pos = get_3d_point(learner_pop[i, 0], learner_pop[i, 1])
                anims.append(dots[i].animate.move_to(end_pos))

            self.play(*anims, run_time=t_move_learner)
            population = learner_pop

            self.remove(iter_text)

        # ==========================================
        # SECTION 4: KẾT THÚC
        # ==========================================
        self.next_section("Ket thuc")
        self.stop_ambient_camera_rotation()

        # Zoom vào điểm tốt nhất tìm được
        final_fitness = np.array([rastrigin(p[0], p[1]) for p in population])
        best_final_idx = np.argmin(final_fitness)
        best_pos_3d = get_3d_point(
            population[best_final_idx, 0], population[best_final_idx, 1]
        )

        self.move_camera(phi=45 * DEGREES, theta=45 * DEGREES, zoom=1.5, run_time=2)

        # Vòng tròn highlight kết quả
        result_circle = (
            Circle(radius=0.2, color=PURE_GREEN)
            .rotate(PI / 2, axis=RIGHT)
            .move_to(best_pos_3d)
        )
        self.play(Create(result_circle), Flash(dots[best_final_idx]))

        # Thêm text thông báo kết thúc
        end_text = Text("Optimization Complete!", font_size=32, color=GREEN).to_corner(
            UP
        )
        self.add_fixed_in_frame_mobjects(end_text)
        self.play(Write(end_text))

        self.wait(2)
