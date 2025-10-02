#include "Soldier.h"
#include "Gun.h"
#include "iostream"
// using namespace std;
void test()
{
    Soldier zhihao("liuzhihao");
    cout << "There is one soldier named " << zhihao._name << "." << endl;
    zhihao.addGun(new Gun("AK47"));
    cout << "We give her a " << zhihao._ptr_gun->_type << "." << endl;
    zhihao.addBulletToGun(20);
    cout << "She adds " << zhihao._ptr_gun->_bullet_count << " bullet(s) to it." << endl;
    cout << "She fired this gun." << endl;
    zhihao.fire();
}

int main()
{
    test();
    return 0;
}