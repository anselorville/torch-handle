# # 魔法学院
# 多多进⼊了魔法学院学习，学院有 n ⻔不同的魔法课程，每⻔课程都有其独特的属性：
# - power[i] ：学习这⻔课程能提升的魔法强度
# - mana[i] ：学习这⻔课程需要消耗的法⼒值
# - 学院的教学楼有 m 层，每层有不同的环境加成系数 bonus[j] （1 ≤ bonus[j] ≤ 3）。
# - 多多总共有 M 点初始法⼒值。
# **特殊规则：**
# - 顺序学习：多多必须按顺序学习课程（必须先学课程1，再学课程2，以此类推）。
# - 楼层绑定：每⻔课程只能在某⼀层完整学习，不能跨层。
# - 强度加成：在第 j 层学习第 i ⻔课程时，获得的实际魔法强度为 power[i] × bonus[j]。
# - 法⼒消耗：在第 j 层学习第 i ⻔课程时，消耗的实际法⼒值为 mana[i] × bonus[j]。
# - 切换代价：多多可以在不同楼层之间切换课程，第⼀次学习选择楼层没有切换代价, 但每次切换可能会额外消耗楼层⾼度差的法⼒值。如果从低楼层切换到⾼楼层, ⽐如从1层切换到4层, 消耗3点法⼒, 如果从⾼楼层切换到低楼层, 则不会消耗额外的法⼒。
# 请求出在满⾜法⼒值限制（总法⼒消耗不超过 M ）的条件下，多多能获得的最⼤魔法强度总和(⽆需学完所有课程)。
# **输⼊描述**
# 第⼀⾏三个整数 n , m , M （1 ≤ n ≤ 100, 1 ≤ m ≤ 5, 1 ≤ M ≤ 1000） 
# 第⼆⾏ n 个整数，表示 power[i] （1 ≤ power[i] ≤ 100） 
# 第三⾏ n 个整数，表示 mana[i] （1 ≤ mana[i] ≤ 100） 
# 第四⾏ m 个整数，表示 bonus[j] （1 ≤ bonus[j] ≤ 3）
# **输出描述**
# 输出⼀个整数，表示能获得的最⼤魔法强度总和。如果⽆法完成任何课程（例如，第⼀⻔课程在任何⼀层学习的法⼒消耗都超过 M ），则输出 0。
# **补充说明**
# - 对于 20% 的数据：n ≤ 10, 1 ≤ m ≤ 5, 1 ≤ M ≤ 1000
# - 对于 60% 的数据：n ≤ 30, 1 ≤ m ≤ 5, 1 ≤ M ≤ 1000
# - 对于 100% 的数据：1 ≤ n ≤ 100, 1 ≤ m ≤ 5, 1 ≤ M ≤ 1000


# max(bonus[j] * power[i]) for i in range(n), j in rang(m)
# sum(mana[i]) < M

def solve(p_list:list[int], m_list:list[int], b_list:list[int], init_m:int):
    pass


if __name__ == "__main__":
    shapes = input("shapes:").split(' ')
    assert len(shapes) == 3, "输入形状参数数量错误"
    n,m,init_m = [int(e.strip()) for e in shapes]
    p_list = input("plist:").split(' ')
    p_list = [int(e.strip()) for e in p_list]
    assert len(p_list) == n, "输入n_list数量错误"
    m_list = input("mlist:").split(' ')
    m_list = [int(e.strip()) for e in m_list]
    assert len(m_list) == n, "输入m_list数量错误"
    b_list = input("blist:").split(' ')
    b_list = [int(e.strip()) for e in b_list]
    assert len(b_list) == m, "输入b_list数量错误"
    score = solve(p_list,m_list,b_list,init_m)





