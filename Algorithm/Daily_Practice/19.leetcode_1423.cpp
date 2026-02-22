class Solution {
    public:
        int maxScore(vector<int>& cardPoints, int k) {
            int n=cardPoints.size();
            int m=n-k;
            int s=reduce(cardPoints.begin(),cardPoints.begin()+m);
            int ans=s;
            for(int i=m;i<n;++i){
                s+=cardPoints[i]-cardPoints[i-m];
                ans=min(s,ans);
            }
            return reduce(cardPoints.begin(),cardPoints.begin()+n)-ans;
        }
    };