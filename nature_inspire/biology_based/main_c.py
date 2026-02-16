import matplotlib.pyplot as plt
from fa import FA
from cs import CS

TEST_FUNC = 'rosenbrock'
DIMENSION = 10
ITERATIONS = 100

print(f"=== SO SÁNH TRÊN HÀM {TEST_FUNC.upper()} ===")

fa = FA(func_name=TEST_FUNC, pop_size=30, dim=DIMENSION, max_iter=ITERATIONS)
fa_best, fa_score = fa.solve()

cs = CS(func_name=TEST_FUNC, pop_size=30, dim=DIMENSION, max_iter=ITERATIONS)
cs_best, cs_score = cs.solve()

print("\n=== KẾT QUẢ CUỐI CÙNG ===")
print(f"Firefly Algorithm: {fa_score}")
print(f"Cuckoo Search:     {cs_score}")

plt.figure(figsize=(10, 6))
plt.plot(fa.history, label='Firefly Algorithm (FA)', color='red', linestyle='--')
plt.plot(cs.history, label='Cuckoo Search (CS)', color='blue')

plt.title(f"So sánh tốc độ hội tụ trên hàm {TEST_FUNC}")
plt.xlabel("Số vòng lặp")
plt.ylabel("Fitness (Log Scale)")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()