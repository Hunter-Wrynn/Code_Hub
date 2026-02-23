class Solution {
    public:
        int distinctPoints(string s, int k) {
            int n = s.size();
            unordered_set<long long> st;
            int x = 0, y = 0;
            for (int i = 0; i < n; i++) {
                switch (s[i]) {
                    case 'L': x--; break;
                    case 'R': x++; break;
                    case 'D': y--; break;
                    case 'U': y++; break;
                }
                int left = i + 1 - k;
                if (left < 0) continue;
                st.insert((long long)(x + n) << 32 | (y + n));
                switch (s[left]) {
                    case 'L': x++; break;
                    case 'R': x--; break;
                    case 'D': y++; break;
                    case 'U': y--; break;
                }
            }
            return st.size();
        }
    };
