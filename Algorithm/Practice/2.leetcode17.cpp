#include<bit/stdc++.h>
using namespace std;

string tmp;
vector<string> res;
vector<string> board = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

void DFS(int pos, const string& digits, string& tmp, vector<string>& res, const vector<string>& board) {
    if (pos == digits.size()) {
        res.push_back(tmp);
        return;
    }
    int num = digits[pos] - '0';
    for (char c : board[num]) {
        tmp.push_back(c);
        DFS(pos + 1, digits, tmp, res, board);
        tmp.pop_back();
    }
}

int main() {
    string digits;

    cin >> digits;

    vector<string> res;
    if (!digits.empty()) {
        vector<string> board = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        string tmp;
        DFS(0, digits, tmp, res, board);
    }


    for (const string& combination : res) {
        cout << combination << endl;
    }

    return 0;
}