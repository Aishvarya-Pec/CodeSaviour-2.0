// EXPECTED_ISSUES: 4

#include <iostream>
int* leak3() { int* p = new int[10]; p[20] = 5; return p; } // buffer overflow + leak
int main() {
    int *p = nullptr; std::cout << *p; // segmentation fault
    int a = 1/0; // divide by zero (UB)
    int arr[2]; arr[5] = 3; // out of bounds
    return 0
}
