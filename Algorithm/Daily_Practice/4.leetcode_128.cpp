class Solution {
    public:
        int longestConsecutive(vector<int>& nums) {
            unordered_set<int> st;
            for(auto num:nums) st.insert(num);
    
            int fans=0;
    
            for(const int&num:st){
                if(!st.count(num-1)){
                    int curnum=num;
                    int curans=1;
    
                    while(st.count(curnum+1)){
                        curnum++;
                        curans++;
                    }
                    
                    fans=max(curans,fans);
    
                }
    
            }
            return fans;
            
        }
    };