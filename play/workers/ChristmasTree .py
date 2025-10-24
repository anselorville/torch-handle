# # 圣诞树 英文：Christmas Tree 
# 圣诞节快到了，有⼀棵挂满彩灯的⼆叉树，需要你来按照图纸装饰。彩灯有5种颜⾊变化，分别⽤1-5表示。1表示 红⾊， 2表示⻩⾊， 3表示蓝⾊， 4表示紫⾊， 5 表示绿⾊。每个节点都⼀个颜⾊控制器，每按⼀下都会产⽣⼀
# 个控制信号。控制信号将从当前节点出发向下传递，将当前节点的彩灯以及以当前节点为根节点的⼦树上的所有节点，切换到下⼀个颜⾊（ 红 -> ⻩-> 蓝 -> 紫 -> 绿 -> 红 ...） 循环切换。
# 给定⼆叉树的初始状态 initial 和 ⽬标状态 target,两者都以**层序遍历**产出的⼀维数组表示。数组元素对应对应位置节点的颜⾊，0表示该节点没有彩灯。
# 请给出从initial状态切换⾄target状态需要的最少控制器点击次数。
# **注意：**
# 1. 控制器按⼀下所产⽣的控制信号，不只影响当前节点，也会影响以当前节点为根节点的⼦树上所有节点切换到下⼀个颜⾊（最终不⼀定是同⼀个颜⾊）。
# 2. 特别地，假设⼦树上的某个节点X上没有彩灯，则祖先节点处发出的控制信号将不会继续传递给X的后代节点。
# **输⼊描述**
# - 第⼀⾏输⼊为⼀个整数n, 代表inital 和 target 数组的⼤⼩。
# - 第⼆⾏输⼊为n个整数，代表inital数组。
# - 第三⾏输⼊为n个整数，代表target数组。
# **其他：**
# - 如果 initial[i] == 0, 则 target[i] 也⼀定为0。
# - 1 <=initial.length <= 106
# **输出描述**
# ⼀个整数，表示最少点击次数

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def switch(self):
        if self.val:
            self.val = self.val + 1 if self.val < 5 else 1
            if self.left:
                self.left.switch()
            if self.right:
                self.right.switch()

    def get_vals(self):
        vals = [self.val]
        if self.left:
            vals.extend(self.left.get_vals())
        if self.right:
            vals.extend(self.right.get_vals())
        return vals


def build_tree(arr):
    """根据层序遍历数组构建二叉树"""
    if not arr or arr[0] == 0:
        return None
    
    root = TreeNode(arr[0])
    queue = [root]
    i = 1
    
    while queue and i < len(arr):
        node = queue.pop(0)
        
        # 左子节点
        if i < len(arr):
            if arr[i] != 0:
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            else:
                node.left = TreeNode(0)  # 无彩灯节点也需要创建
            i += 1
        
        # 右子节点
        if i < len(arr):
            if arr[i] != 0:
                node.right = TreeNode(arr[i])
                queue.append(node.right)
            else:
                node.right = TreeNode(0)
            i += 1
    
    return root


def get_color_diff(current, target):
    """计算从current颜色切换到target颜色需要的次数"""
    if current == 0 or target == 0:
        return 0
    
    diff = (target - current) % 5
    return diff


def solve(initial, target):
    """求解最少点击次数"""
    if not initial or not target:
        return 0
    
    n = len(initial)
    clicks = 0
    
    # 记录每个节点被祖先累计切换的次数
    parent_clicks = [0] * n
    
    # 层序遍历处理每个节点
    for i in range(n):
        if initial[i] == 0:
            continue
        
        # 计算当前节点经过祖先影响后的实际颜色
        actual_color = initial[i]
        if parent_clicks[i] > 0:
            actual_color = ((initial[i] - 1 + parent_clicks[i]) % 5) + 1
        
        # 计算当前节点需要额外点击的次数
        need_clicks = get_color_diff(actual_color, target[i])
        
        if need_clicks > 0:
            clicks += need_clicks
            # 更新子节点被影响的次数
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            
            if left_idx < n:
                parent_clicks[left_idx] += need_clicks
            if right_idx < n:
                parent_clicks[right_idx] += need_clicks
    
    return clicks


if __name__ == "__main__":
    # 读取输入
    n = int(input())
    initial = list(map(int, input().split()))
    target = list(map(int, input().split()))
    
    # 求解并输出结果
    result = solve(initial, target)
    print(result)
    
    # 测试样例
    print("\n=== 测试样例 ===")
    
    # 样例1：简单切换
    test_initial = [1, 2, 3, 0, 0, 0, 0]
    test_target = [2, 3, 4, 0, 0, 0, 0]
    print(f"测试1: initial={test_initial}, target={test_target}")
    print(f"结果: {solve(test_initial, test_target)} (预期: 1，点击根节点一次)")
    
    # 样例2：复杂切换
    test_initial = [1, 2, 3, 4, 5, 0, 0]
    test_target = [3, 3, 4, 5, 1, 0, 0]
    print(f"\n测试2: initial={test_initial}, target={test_target}")
    print(f"结果: {solve(test_initial, test_target)}")
    
    # 样例3：包含无彩灯节点
    test_initial = [1, 2, 0, 4, 5, 0, 0]
    test_target = [2, 3, 0, 5, 1, 0, 0]
    print(f"\n测试3: initial={test_initial}, target={test_target}")
    print(f"结果: {solve(test_initial, test_target)}")

