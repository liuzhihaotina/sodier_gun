#include "Soldier.h"

Soldier::Soldier(string name)
{
    this->_name = name;
    this->_ptr_gun = nullptr; // 初始化为空指针
}

void Soldier::addGun(Gun *ptr_gun)
{
    this->_ptr_gun = ptr_gun;
}

void Soldier::addBulletToGun(int num)
{
    this->_ptr_gun->addBullet(num);
}

bool Soldier::fire()
{
    return this->_ptr_gun->shoot();
}

Soldier::~Soldier()
{
    if (this->_ptr_gun == nullptr)
    {
        return;
    }
    delete this->_ptr_gun;
    this->_ptr_gun = nullptr;
}