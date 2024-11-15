#include <bits/stdc++.h>
using namespace std;

int a[101];
inline int read();
void quicksort(int left, int right);

int main() {
    int n = read();
    for (int i = 1; i <= n; ++i) {  
        a[i] = read();
    }
    quicksort(1, n);
    for (int i = 1; i <= n; i++) {
        cout << a[i] << ' ';
    }
    return 0;
}

inline int read() {
    int x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}

void quicksort(int left, int right) {
    if (left >= right) return;
    int temp = a[left];
    int i = left;
    int j = right;
    int t;

    while (i < j) {
        while (i < j && a[j] >= temp) {
            j--;
        }
        while (i < j && a[i] <= temp) {
            i++;
        }
        if (i < j) {
            t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    a[left] = a[i];
    a[i] = temp;

    quicksort(left, i - 1);
    quicksort(i + 1, right); 
}
