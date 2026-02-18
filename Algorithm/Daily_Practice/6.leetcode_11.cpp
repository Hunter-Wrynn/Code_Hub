class Solution {
    public:
        int maxArea(vector<int>& height) {
            int left=0,right=height.size()-1;
            int ans=0;
    
            while(left<right){
                int cur=min(height[left],height[right])*(right-left);   
                ans=max(ans,cur);
                if(height[left]<height[right]) left++;
                else right--;
            }
            return ans;
        }
    };