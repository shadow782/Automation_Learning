st = []
for i,x in enumerate(nums):
    while st and x < nums[st[-1]]:
        st.pop()


def clearDigits(s: str) -> str:
    st = []
    for c in s:
        if c.isdigit():
            st.pop()
        else:
            st.append(c)
    return st 