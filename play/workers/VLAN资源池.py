# # 题目描述
# [VLAN]是一种对局域网设备进行逻辑划分的技术，为了标识不同的VLAN，引入VLAN ID(1-4094之间的整数)的概念。
# 定义一个VLAN ID的资源池(下称VLAN资源池)，资源池中连续的VLAN用开始VLAN-结束VLAN表示，不连续的用单个整数表示，所有的VLAN用英文逗号连接起来。
# 现在有一个VLAN资源池，业务需要从资源池中申请一个VLAN，需要你输出从VLAN资源池中移除申请的VLAN后的资源池。
# ## 输入描述
# 第一行为字符串格式的VLAN资源池，第二行为业务要申请的VLAN，VLAN的取值范围为[1,4094]之间的整数。
# ## 输出描述
# 从输入VLAN资源池中移除申请的VLAN后字符串格式的VLAN资源池，输出要求满足题目描述中的格式，并且按照VLAN从小到大升序输出。  
# 如果申请的VLAN不在原VLAN资源池内，输出原VLAN资源池升序排序后的字符串即可。
# ## 示例1
# 输入
#     1-5
#     2
# 输出
#     1,3-5
# 说明
# > 原VLAN资源池中有VLAN 1、2、3、4、5，从资源池中移除2后，剩下VLAN 1、3、4、5，按照题目描述格式并升序后的结果为1,3-5
# ## 示例2
# 输入
#     20-21,15,18,30,5-10
#     15
# 输出
#     5-10,18,20-21,30
# 说明
# > 原VLAN资源池中有VLAN 5、6、7、8、9、10、15、18、20、21、30，从资源池中移除15后，资源池中剩下的VLAN为 5、6、7、8、9、10、18、20、21、30，按照题目描述格式并升序后的结果为5-10,18,20-21,30。
# ## 示例3
# 输入
#     5,1-3
#     10
# 输出
#     1-3,5
    
# 说明
# > 原VLAN资源池中有VLAN 1、2、3，5，申请的VLAN 10不在原资源池中，将原资源池按照题目描述格式并按升序排序后输出的结果为1-3,5。

def extract_num(bind_str:str):
    raw_head_tail = bind_str.split('-')
    if len(raw_head_tail) != 2:
        print("输入的合并资源字符串格式不符合“开始VLAN-结束VLAN”")
        return []
    head,tail = int(raw_head_tail[0]),int(raw_head_tail[1])
    return [e for e in range(head,tail+1)]

def reform_num(num_list:list[int]):
    num_list.sort()
    return str(num_list[0]) if len(num_list)==1 else f'{num_list[0]}-{num_list[-1]}' 

def vlan_processor(resource:str,select:str):
    raw_list = [e.strip() for e in resource.split(",")]
    clean_list = []
    for raw_part in raw_list:
        if '-' in raw_part:
            ext_res = extract_num(raw_part)
            clean_list.extend(ext_res)
        else:
            clean_list.append(int(raw_part))
    
    clean_list.sort()
    print(clean_list)
    prev = clean_list[0] - 1
    res_list, tmp_list = [],[]
    
    for num in clean_list:
        if num != int(select.strip()) and (num - prev) != 1 or num == int(select.strip()):
            if tmp_list:
                res_list.append(reform_num(tmp_list))
                tmp_list = []

        if num != int(select.strip()):
            tmp_list.append(num)
        # 更新标记
        prev = num

    res_list.append(reform_num(tmp_list))

    print(','.join(res_list))
    return res_list 
                
if __name__ == "__main__":
    res = input("请输入VLAN资源:")
    sel = input("请输入申请的VLAN:")
    vlan_processor(res,sel)