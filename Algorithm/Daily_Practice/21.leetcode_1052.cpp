class Solution {
    public:
        int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int minutes) {
            int cnt=0;
            int a=0;
            int ans=0;
            for (int i=0;i<customers.size();++i){
                if(grumpy[i]==0) a+=customers[i];
                if(grumpy[i]==1){
                    cnt+=customers[i];
                }
                if(i-minutes+1<0) continue;
                ans=max(cnt,ans);
                if(grumpy[i-minutes+1]==1) cnt-=customers[i-minutes+1];
            }
            return a+ans;
        }
    };