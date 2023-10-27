import numpy as np


def test_silver():
    n = 9
    rho = 1 + np.sqrt(2)
    silver_steps = [1 + rho ** ((k & -k).bit_length()-2) for k in range(1, n+1)]
    print(silver_steps)


def main():
    test_silver()


if __name__ == '__main__':
    main()
