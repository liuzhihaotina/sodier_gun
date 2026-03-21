#include <iostream>

int main()
{
    int y { 5 }; // 在这里定义y，因为在外围的代码块中需要使用y

    {
        int x{};
        std::cin >> x;

        // 如果在这个y的第一次使用之前定义y，在外围块中将无法访问到y
        if (x == 4)
            y = 4;
    }

    std::cout << y; // 需要在这里能访问到y

    return 0;
}