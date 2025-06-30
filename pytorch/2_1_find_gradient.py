import torch


# 2.6
def find_2_6():
    ten = torch.tensor([[1., 2.], [4., 5.]], requires_grad=True)  # Превращает тензор в переменную по которой можно считать произовдную
    function = 10 * torch.log(ten + 1.).sum()
    function.backward()  # Вычисление значения градиентного спуска
    print(ten-ten.grad)


# 2.7
def find_2_7_1():
    w = torch.tensor([[5, 10], [1, 2]], requires_grad=True, dtype=torch.float32)
    function = torch.log(torch.log(w + 7)).prod()
    function.backward() 
    # print(w.grad) # Код для самопроверки


def find_2_7_2():
    w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
    alpha = 0.001

    for _ in range(500):
        function = (w + 7).log().log().prod()
        function.backward()
        w.data -= alpha * w.grad
        w.grad.zero_()
    # print(w)



find_2_7_2()
