"""
圣诞树彩灯控制问题 (Christmas Tree)

问题描述：
    有一棵挂满彩灯的二叉树，彩灯有5种颜色：1-红色，2-黄色，3-蓝色，4-紫色，5-绿色
    点击节点的控制器，会让该节点及其子树的所有彩灯切换到下一个颜色（循环：1→2→3→4→5→1）
    
关键规则：
    1. 点击影响：当前节点 + 整个子树（如果子节点有彩灯）
    2. 信号阻断：如果节点值为0（无彩灯），信号不会传递到其子节点
    3. 给定初始状态和目标状态（层序遍历数组），求最少点击次数

输入：
    - n: 数组大小
    - initial: 初始状态数组（层序遍历）
    - target: 目标状态数组（层序遍历）
    - 0表示该位置无彩灯
    - 1 <= n <= 10^6

输出：
    - 最少点击次数

算法核心：
    贪心算法 + 自顶向下处理
    时间复杂度：O(n)
    空间复杂度：O(n)
"""


def get_color_diff(current, target):
    """
    计算从current颜色切换到target颜色需要的最少点击次数
    
    Args:
        current: 当前颜色值 (0-5)
        target: 目标颜色值 (0-5)
    
    Returns:
        需要点击的次数 (0-4)
    
    示例：
        1 -> 3: 需要2次点击 (1->2->3)
        5 -> 2: 需要2次点击 (5->1->2)
        3 -> 3: 需要0次点击
    """
    # 如果当前或目标为0（无彩灯），无需点击
    if current == 0 or target == 0:
        return 0
    
    # 如果颜色相同，无需点击
    if current == target:
        return 0
    
    # 计算颜色差值（循环）
    # 使用模运算处理循环：(target - current) % 5
    # 例如：5 -> 2: (2 - 5) % 5 = -3 % 5 = 2
    diff = (target - current) % 5
    return diff


def solve(initial, target, debug=False):
    """
    求解圣诞树彩灯问题的最少点击次数
    
    核心思路：
        1. 使用贪心策略，自顶向下处理每个节点
        2. 维护每个节点被祖先累计影响的次数
        3. 对每个节点：计算实际颜色 -> 计算需要的点击次数 -> 传递影响到子节点
        4. 遇到0节点时，不传递影响到其子节点（信号阻断）
    
    为什么贪心正确？
        - 点击只影响自己和子孙，不影响祖先和兄弟
        - 自顶向下处理时，每个节点的祖先影响已确定
        - 因此每个节点可以独立决策，局部最优 = 全局最优
    
    Args:
        initial: 初始颜色数组（层序遍历）
        target: 目标颜色数组（层序遍历）
        debug: 是否输出调试信息
    
    Returns:
        最少点击次数
    """
    # 边界条件检查
    if not initial or not target:
        return 0
    
    if len(initial) != len(target):
        return -1  # 输入不合法
    
    n = len(initial)
    total_clicks = 0
    
    # parent_clicks[i]: 节点i被祖先累计点击的次数
    # 用于计算节点的实际颜色
    parent_clicks = [0] * n
    
    if debug:
        print(f"\n【调试模式】开始处理，共 {n} 个节点")
        print(f"初始: {initial}")
        print(f"目标: {target}\n")
    
    # 层序遍历处理每个节点（从根节点开始）
    for i in range(n):
        # 如果当前节点无彩灯，跳过处理
        # 注意：信号阻断 - 0节点不会将祖先的影响传递给子节点
        if initial[i] == 0:
            if debug:
                print(f"节点{i}: 无彩灯(0)，跳过")
            continue
        
        # 步骤1：计算当前节点受祖先影响后的实际颜色
        # 实际颜色 = 初始颜色 + 祖先点击次数（模5循环）
        actual_color = initial[i]
        if parent_clicks[i] > 0:
            # 颜色值1-5，转换为0-4计算，最后再加1
            actual_color = ((initial[i] - 1 + parent_clicks[i]) % 5) + 1
        
        # 步骤2：计算从实际颜色到目标颜色需要的点击次数
        need_clicks = get_color_diff(actual_color, target[i])
        
        if debug:
            print(f"节点{i}: 初始={initial[i]}, 祖先影响={parent_clicks[i]}, "
                  f"实际={actual_color}, 目标={target[i]}, 需点击={need_clicks}")
        
        # 步骤3：计算该节点总共产生的影响次数（祖先影响 + 自己的点击）
        # 关键：即使自己不需要点击，也要将祖先的影响传递给子节点！
        total_影响 = parent_clicks[i] + need_clicks
        
        # 更新总点击数
        if need_clicks > 0:
            total_clicks += need_clicks
        
        # 步骤4：将累积的影响传递给子节点
        # 只要当前节点有彩灯（不是0），就应该传递影响
        if total_影响 > 0:
            left_idx = 2 * i + 1   # 左子节点
            right_idx = 2 * i + 2  # 右子节点
            
            if left_idx < n:
                parent_clicks[left_idx] += total_影响
                if debug:
                    print(f"  -> 传递到左子节点{left_idx}: +{total_影响}")
            
            if right_idx < n:
                parent_clicks[right_idx] += total_影响
                if debug:
                    print(f"  -> 传递到右子节点{right_idx}: +{total_影响}")
    
    if debug:
        print(f"\n总点击次数: {total_clicks}\n")
    
    return total_clicks


def run_test(test_name, initial, target, expected=None):
    """
    运行单个测试样例
    
    Args:
        test_name: 测试名称
        initial: 初始状态
        target: 目标状态
        expected: 预期结果（可选）
    """
    result = solve(initial, target)
    print(f"\n{test_name}")
    print(f"  初始状态: {initial}")
    print(f"  目标状态: {target}")
    print(f"  最少点击: {result}", end="")
    if expected is not None:
        status = "✓" if result == expected else "✗"
        print(f" (预期: {expected}) {status}")
    else:
        print()
    return result


def visualize_tree(arr, name="树结构"):
    """
    可视化二叉树（简单格式）
    
    Args:
        arr: 层序遍历数组
        name: 树的名称
    """
    print(f"\n{name}:")
    if not arr:
        print("  空树")
        return
    
    # 简单的层次显示
    level = 0
    idx = 0
    while idx < len(arr):
        nodes_in_level = 2 ** level
        level_nodes = arr[idx:idx + nodes_in_level]
        indent = "  " * (4 - level) if level < 4 else ""
        print(f"  {indent}Level {level}: {level_nodes}")
        idx += nodes_in_level
        level += 1
        if idx >= len(arr):
            break


if __name__ == "__main__":
    import sys
    import os
    
    # 判断是否需要在线测试模式
    # 使用环境变量或命令行参数控制
    # 用法：ONLINE_JUDGE=1 python ChristmasTree.py  或  python ChristmasTree.py --online
    online_mode = (
        len(sys.argv) > 1 and sys.argv[1] == '--online'
    ) or (
        'ONLINE_JUDGE' in os.environ or 'OJ' in os.environ
    )
    
    if online_mode:
        # 在线测试模式：读取标准输入
        n = int(input())
        initial = list(map(int, input().split()))
        target = list(map(int, input().split()))
        
        # 求解并输出结果
        result = solve(initial, target)
        print(result)
    else:
        # 本地测试模式：运行测试样例
        print("=" * 60)
        print("圣诞树彩灯控制问题 - 测试样例")
        print("=" * 60)
        
        # 样例1：简单切换 - 点击根节点一次
        # 树结构：    1
        #          /   \
        #         2     3
        # 点击根节点1次：所有节点+1 -> [2, 3, 4]
        run_test(
            "测试1: 简单切换（点击根节点）",
            initial=[1, 2, 3, 0, 0, 0, 0],
            target=[2, 3, 4, 0, 0, 0, 0],
            expected=1
        )
        
        # 样例2：信号阻断测试
        # 树结构：    1
        #          /   \
        #         2     0  <-- 无彩灯，阻断信号
        #        / \
        #       4   5
        # 点击根节点1次：节点1的子树全部+1，节点2(0)的子树不受影响
        run_test(
            "测试2: 信号阻断（0节点）",
            initial=[1, 2, 0, 4, 5, 0, 0],
            target=[2, 3, 0, 5, 1, 0, 0],
            expected=1
        )
        
        # 样例3：全部相同，无需点击
        run_test(
            "测试3: 已达目标（无需点击）",
            initial=[1, 2, 3, 4, 5],
            target=[1, 2, 3, 4, 5],
            expected=0
        )
        
        # 样例4：单节点切换
        run_test(
            "测试4: 单节点（1→3需要2次）",
            initial=[1],
            target=[3],
            expected=2
        )
        
        # 样例5：循环切换（5->1->2）
        run_test(
            "测试5: 循环切换（5→1→2）",
            initial=[5, 4, 3],
            target=[2, 1, 5],
            expected=2
        )
        
        # 样例6：深度树，所有节点统一变化
        # 15个节点组成的完全二叉树，全部从1变到2
        # 只需点击根节点1次，影响传递到所有子孙
        run_test(
            "测试6: 深度树（一次点击传递到所有节点）",
            initial=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            target=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            expected=1
        )
        
        # 样例7：复杂场景 - 不同节点需要不同的点击次数
        # 树结构：    1
        #          /   \
        #         2     3
        #        / \
        #       4   5
        # 需要精细控制每个节点
        run_test(
            "测试7: 复杂切换（多节点独立调整）",
            initial=[1, 2, 3, 4, 5, 0, 0],
            target=[3, 3, 4, 5, 1, 0, 0]
        )
        
        # 样例8：大步长循环
        run_test(
            "测试8: 大步长（1→5需要4次）",
            initial=[1, 2, 3],
            target=[5, 1, 2],
            expected=4
        )
        
        # 样例9：多个0节点阻断
        # 树结构：      1
        #            /     \
        #          0       0  <- 两个子节点都无彩灯，阻断信号
        #         / \     / \
        #        2   3   4   5
        # 根节点点击后，信号被阻断，无法传递到叶子节点
        # 需要：根(1) + 节点3(1) + 节点4(1) + 节点5(1) + 节点6(1) = 5次
        run_test(
            "测试9: 多处信号阻断（需分别处理）",
            initial=[1, 0, 0, 2, 3, 4, 5],
            target=[2, 0, 0, 3, 4, 5, 1],
            expected=5
        )
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

