from src.agent.agent_L25 import __inner_L25
from src.agent.agent_L25 import *


def main():
    test_l25 = nx_graph({'W', 'X', 'Y', 'Z'},
                          {('W', 'X'), ('W', 'Y'),
                           ('X', 'Y'), ('X', 'Z'),
                           ('Z', 'Y')
                           },
                          {},
                          ordered_topology=('W', 'X', 'Z', 'Y'),
                          )

    def compute_naive_l25():
        # L2.5 실험 한 번 (SCM 하나 + 샘플링들) 수행
        return __inner_L25(test_l25)

    results_MSE = []
    for i in range(10):
        num_iterations = 100
        results = Parallel(n_jobs=-1)(
            delayed(compute_naive_l25)() for _ in tqdm(range(num_iterations))
        )

        # __inner_L25가 (MSE_naive, MSE_L25)를 리턴
        MSE_naive_list = [res[0] for res in results]
        MSE_L25_list  = [res[1] for res in results]

        MSE_naive = np.mean(MSE_naive_list)
        MSE_L25   = np.mean(MSE_L25_list)

        reduction_ratio = (MSE_naive - MSE_L25) / MSE_naive * 100

        print(f"[iter {i}] MSE_naive: {MSE_naive:.6f}, "
              f"MSE_L25: {MSE_L25:.6f}, diff: {MSE_naive - MSE_L25:.6f}")
        print(f"Reduction ratio: {reduction_ratio:.5f}%")

        results_MSE.append(reduction_ratio)

    print("================")
    results_MSE = np.array(results_MSE)
    # print(results_MSE)
    print("max reduction:", results_MSE.max())
    print("mean reduction:", results_MSE.mean())

if __name__ == '__main__':
    main()