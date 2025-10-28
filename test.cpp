#include<iostream>
using namespace std;
int main() {
    char char1 = 'A';
    int int_char1 = int(char1);
    cout << int_char1 << endl;
    int int_char2 = 66;
    char char2 = char(int_char2);
    cout << char2 << endl;
    return 0;
}