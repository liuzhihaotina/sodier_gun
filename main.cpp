#include <iostream>
#include "test1.h"
#include "test2.h"
using namespace std;

int main()
{
    int a = 0;
    cout << "Hello, World! \n\\" << a << endl;
    cout << "Use task.json to \tbuild." << endl;
    test1();
    test2();
    return 0;
}