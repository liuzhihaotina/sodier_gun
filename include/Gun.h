#pragma once // 防止当前头文件被重复包含
#include <string>

using namespace std; // 大多数函数和对象都定义在std命名空间

class Gun
{
public:
    Gun(string type)
    {
        this->_bullet_count = 0;
        this->_type = type;
    }
    string _type;
    int _bullet_count;
    // ~Gun();
    void addBullet(int bullet_num);
    bool shoot();

private:
    // int _bullet_count;
    // string _type;
};