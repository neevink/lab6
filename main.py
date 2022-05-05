import math
from collections import namedtuple

from methods import AdamsMethod, ImprovedEulerMethod


Hint = namedtuple('Hint', ['a', 'b', 'y0', 'h', 'e'])
Function = namedtuple('Function', ['desc', 'func', 'reference', 'calc_const', 'hint'])

functions = [
    Function(
        'y` = y + (1 + x) * y^2',
        lambda x, y: y + (1 + x) * y ** 2,
        lambda x, c: - math.e ** x / (x * math.e ** x + (math.e - math.e)),
        lambda x0, y0: -math.e ** x0 / y0 - x0 * math.e ** x0,
        Hint(1, 2, -1, 0.05, 0.01),
    ),
    Function(
        'y` = (x + 1)^3 - y',
        lambda x, y: (x + 1) ** 3 - y,
        lambda x, c: c * math.e ** (-x) + x**3 + 3*x - 2,
        lambda x0, y0: (y0 - x0**3 - 3*x0 + 2) * math.e ** x0,
        Hint(0, 3, 0, 0.1, 0.0001),
    ),
    Function(
        'y` = xy',
        lambda x, y: x * y,
        lambda x, c: c * math.e ** (x**2/2),
        lambda x0, y0: y0 / (math.e ** (x0**2 / 2)),
        Hint(-1, 1, 1, 0.01, 0.0001),
    ),
]

methods = [
    ImprovedEulerMethod(),
    AdamsMethod(),
]


def main():
    print('Выберите уравнение для задачи Коши:')
    for i, f in enumerate(functions, 1):
        print(f'{i}. {f.desc}')
    func = functions[int(input()) - 1]

    a, b = map(float, input(f'Интервал дифференцирования [a; b] ({func.hint.a} {func.hint.b}): ').split())
    y0 = float(input(f'Начальные условия y0 ({func.hint.y0}): '))
    h = float(input(f'Шаг дифференцирования ({func.hint.h}): '))
    e = float(input(f'Точность e ({func.hint.e}): '))

    for method in methods:
        print('Решение задачи Коши методом:', method.get_name())
        method.solve(
            func.func,
            func.reference,
            func.calc_const(a, y0),
            a,
            b,
            y0,
            h,
            e,
        )
        print()


if __name__ == "__main__":
    main()
