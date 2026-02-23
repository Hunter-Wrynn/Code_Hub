class Solution {
    public:
        int maxFreeTime(int eventTime, int k, vector<int>& startTime, vector<int>& endTime) {
            vector<int> ans;
            ans.push_back(startTime[0]);
            for(int i=0;i<endTime.size()-1;++i){
                ans.push_back(startTime[i+1]-endTime[i]);
            }
            ans.push_back(eventTime-endTime[endTime.size()-1]);
            int cnt=0;
            int as=0;
            for(int i=0;i<ans.size();++i){
                cnt+=ans[i];
                if(i-k<0) continue;
                as=max(cnt,as);
                cnt-=ans[i-k];
            }
            return as;
        }
    };
