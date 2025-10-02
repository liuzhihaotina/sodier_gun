// #include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>

#include "include/Gun.h"
#include "include/Soldier.h"

namespace py = pybind11;

PYBIND11_MODULE(soldier_gun, m) { // 模块名，该模块含有多个类
    py::class_<Soldier>(m, "Soldier")  // < >内是cpp的类名，" "里是暴露给python的别名，python调用的类名
        .def(py::init<std::string>())  
        .def_readwrite("_name", &Soldier::_name)  
        .def("addGun", &Soldier::addGun)
        .def("addBulletToGun", &Soldier::addBulletToGun)
        .def("fire", &Soldier::fire);
        // .def("Soldier_release", &Soldier::~Soldier) // pybind11会自动管理内存。这里为了在Python中手动触发析构，但一般不推荐这样做
    
    py::class_<Gun>(m, "Gun")
        .def(py::init<std::string>())
        .def_readwrite("_bullet_count", &Gun::_bullet_count) // 暴露 _bullet_count 成员变量
        .def_readwrite("_type", &Gun::_type)
        .def("addBullet", &Gun::addBullet)
        .def("shoot", &Gun::shoot);
}

