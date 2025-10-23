## 题目描述
[VLAN](https://so.csdn.net/so/search?q=VLAN&spm=1001.2101.3001.7020)是一种对局域网设备进行逻辑划分的技术，为了标识不同的VLAN，引入VLAN ID(1-4094之间的整数)的概念。
定义一个VLAN ID的资源池(下称VLAN资源池)，资源池中连续的VLAN用开始VLAN-结束VLAN表示，不连续的用单个整数表示，所有的VLAN用英文逗号连接起来。
现在有一个VLAN资源池，业务需要从资源池中申请一个VLAN，需要你输出从VLAN资源池中移除申请的VLAN后的资源池。
## 输入描述
第一行为字符串格式的VLAN资源池，第二行为业务要申请的VLAN，VLAN的取值范围为[1,4094]之间的整数。
## 输出描述
从输入VLAN资源池中移除申请的VLAN后字符串格式的VLAN资源池，输出要求满足题目描述中的格式，并且按照VLAN从小到大升序输出。  
如果申请的VLAN不在原VLAN资源池内，输出原VLAN资源池升序排序后的字符串即可。
## 示例1
输入
    1-5
    2
输出
    1,3-5
说明
> 原VLAN资源池中有VLAN 1、2、3、4、5，从资源池中移除2后，剩下VLAN 1、3、4、5，按照题目描述格式并升序后的结果为1,3-5
## 示例2
输入
    20-21,15,18,30,5-10
    15
输出
    5-10,18,20-21,30
说明
> 原VLAN资源池中有VLAN 5、6、7、8、9、10、15、18、20、21、30，从资源池中移除15后，资源池中剩下的VLAN为 5、6、7、8、9、10、18、20、21、30，按照题目描述格式并升序后的结果为5-10,18,20-21,30。
## 示例3
输入
    5,1-3
    10
输出
    1-3,5
    
说明
> 原VLAN资源池中有VLAN 1、2、3，5，申请的VLAN 10不在原资源池中，将原资源池按照题目描述格式并按升序排序后输出的结果为1-3,5。
## 解题思路
  * **VLAN** （[虚拟局域网](https://so.csdn.net/so/search?q=%E8%99%9A%E6%8B%9F%E5%B1%80%E5%9F%9F%E7%BD%91&spm=1001.2101.3001.7020)）是一种网络技术，用来对局域网中的设备进行逻辑划分。每个VLAN通过一个 **VLAN ID** 来标识，取值范围是 1 到 4094 的整数。题目中的任务是处理一个字符串格式的 VLAN 资源池，模拟从资源池中申请并移除某个 VLAN，然后返回剩余的资源池。
##### 任务：
    1. 输入包含两个部分：
       * **VLAN资源池** ：用字符串表示，可能是单个 VLAN ID，或者是多个 VLAN ID 或 VLAN ID 范围（用"-"连接），各个 VLAN 或范围之间用逗号连接。
       * **要申请的 VLAN** ：一个需要移除的 VLAN ID。
    2. 输出：
       * 从资源池中移除申请的 VLAN 后，输出剩余的 VLAN 资源池，按从小到大的顺序排列，且格式必须符合题目的描述。
##### 输出格式：
    * **连续的 VLAN ID** 应该用范围的方式表示，如 `1-5` 表示 VLAN 1, 2, 3, 4, 5。
    * **不连续的 VLAN ID** 用逗号分隔，如 `1,3-5`。
#### 示例分析：
##### 示例 1：
输入：
    1-5
    2
解释：
    * 原始资源池有 VLAN 1, 2, 3, 4, 5。
    * 申请的 VLAN 是 2，所以移除 2 后，剩下的 VLAN 是 1, 3, 4, 5。
    * 格式化输出为：`1,3-5`。
##### 示例 2：
输入：
    20-21,15,18,30,5-10
    15
解释：
    * 原始资源池有 VLAN 5, 6, 7, 8, 9, 10, 15, 18, 20, 21, 30。
    * 申请的 VLAN 是 15，移除 15 后，剩下的 VLAN 是 5, 6, 7, 8, 9, 10, 18, 20, 21, 30。
    * 格式化输出为：`5-10,18,20-21,30`。
##### 示例 3：
输入：
    5,1-3
    10
解释：
    * 原始资源池有 VLAN 1, 2, 3, 5。
    * 申请的 VLAN 是 10，不在资源池中，所以资源池保持不变。
    * 格式化输出为：`1-3,5`。
## Java
    
    import java.util.*;
    
    public class Main {
        public static void main(String[] args) {
            Scanner sc = new Scanner(System.in);
    
            // 输入VLAN资源池
            String input = sc.nextLine();
            // 输入业务要申请的VLAN
            Integer destVlan = Integer.parseInt(sc.nextLine());
    
            // 解析VLAN资源池
            List<Integer> vlanPool = parseVlanPool(input);
    
            // 对VLAN资源池进行升序排序
            Collections.sort(vlanPool);
    
            // 从VLAN资源池中移除申请的VLAN
            vlanPool.remove(destVlan);
    
            // 格式化VLAN资源池
            String result = formatVlanPool(vlanPool);
            System.out.println(result);
        }
    
        // 解析VLAN资源池
        private static List<Integer> parseVlanPool(String input) {
            List<Integer> vlanPool = new ArrayList<Integer>();
            // 根据逗号分割VLAN资源池中的VLAN
            String[] vlanGroup = input.split(",");
            for (String vlanItem : vlanGroup) {
                if (vlanItem.contains("-")) {
                    // 如果VLAN是连续的，根据连字符分割开始VLAN和结束VLAN
                    String[] vlanItems = vlanItem.split("-");
                    Integer start = Integer.parseInt(vlanItems[0]);
                    Integer end = Integer.parseInt(vlanItems[1]);
                    // 将连续的VLAN添加到VLAN资源池中
                    for (int j = start; j <= end; j++) {
                        vlanPool.add(j);
                    }
                } else {
                    // 如果VLAN是单个的，直接添加到VLAN资源池中
                    vlanPool.add(Integer.parseInt(vlanItem));
                }
            }
            return vlanPool;
        }
    
        // 格式化VLAN资源池
        private static String formatVlanPool(List<Integer> vlanPool) {
            StringBuilder result = new StringBuilder();
            Integer last = null;
            for (int index = 0; index < vlanPool.size(); index++) {
                if (last == null) {
                    // 如果是第一个VLAN，直接添加到结果中
                    result.append(vlanPool.get(index));
                    last = vlanPool.get(index);
                } else {
                    if (vlanPool.get(index) - last == 1) {
                        // 如果与上一个VLAN相差1，表示是连续的VLAN
                        if (result.toString().endsWith("-" + last)) {
                            // 如果结果中最后一个VLAN已经是连续的VLAN的结束VLAN，替换为当前VLAN
                            result.replace(result.lastIndexOf(last.toString()), result.length(), vlanPool.get(index).toString());
                        } else {
                            // 否则添加连字符和当前VLAN
                            result.append("-").append(vlanPool.get(index));
                        }
                    } else {
                        // 如果与上一个VLAN不连续，直接添加逗号和当前VLAN
                        result.append(",").append(vlanPool.get(index));
                    }
                    last = vlanPool.get(index);
                }
            }
            return result.toString();
        }
    }
    
    
    ![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCodeMoreWhite.png)
## Python
    
    
    import sys
    
    # 输入VLAN资源池
    vlan_pool_input = input()
    # 输入业务要申请的VLAN
    dest_vlan = int(input())
    
    # 定义存储VLAN的列表
    vlan_pool = []
    
    # 将输入的VLAN资源池按逗号分隔为多个VLAN组
    vlan_group = vlan_pool_input.split(",")
    
    # 遍历每个VLAN组
    for vlan_item in vlan_group:
        # 如果VLAN组中包含连续的VLAN
        if "-" in vlan_item:
            # 将连续的VLAN拆分为开始VLAN和结束VLAN
            vlan_items = vlan_item.split("-")
            start_vlan = int(vlan_items[0])
            end_vlan = int(vlan_items[1])
            # 将连续的VLAN添加到VLAN资源池中
            for j in range(start_vlan, end_vlan + 1):
                vlan_pool.append(j)
            continue
        # 如果VLAN组中只有一个VLAN
        vlan_pool.append(int(vlan_item))
    
    # 对VLAN资源池进行升序排序
    vlan_pool.sort()
    
    # 如果申请的VLAN在VLAN资源池中
    if dest_vlan in vlan_pool:
        # 从VLAN资源池中移除申请的VLAN
        vlan_pool.remove(dest_vlan)
    
    # 定义存储结果的列表
    result = []
    # 定义上一个VLAN的变量
    last_vlan = None
    
    # 遍历VLAN资源池中的每个VLAN
    for index in range(len(vlan_pool)):
        # 如果是第一个VLAN
        if last_vlan is None:
            result.append(str(vlan_pool[index]))
            last_vlan = vlan_pool[index]
            continue
        # 如果当前VLAN与上一个VLAN连续
        if vlan_pool[index] - last_vlan == 1:
            # 如果结果列表中的最后一个元素以"-上一个VLAN"结尾
            if result[-1].endswith("-" + str(last_vlan)):
                # 将结果列表中的最后一个元素更新为"-当前VLAN"
                result[-1] = result[-1][:result[-1].rindex(str(last_vlan))] + str(vlan_pool[index])
            else:
                # 在结果列表中添加"-当前VLAN"
                result.append("-" + str(vlan_pool[index]))
        else:
            # 在结果列表中添加",当前VLAN"
            result.append("," + str(vlan_pool[index]))
        last_vlan = vlan_pool[index]
    
    # 输出结果列表中的VLAN资源池
    print("".join(result))
    
    
    ![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCodeMoreWhite.png)
## JavaScript
    
    
    const readline = require('readline');
    
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    
    // 输入VLAN资源池
    rl.on('line', (vlan_pool_input) => {
      // 输入业务要申请的VLAN
      rl.on('line', (dest_vlan_input) => {
        // 关闭读取接口
        rl.close();
    
        const dest_vlan = parseInt(dest_vlan_input);
    
        // 定义存储VLAN的列表
        const vlan_pool = [];
    
        // 将输入的VLAN资源池按逗号分隔为多个VLAN组
        const vlan_group = vlan_pool_input.split(",");
    
        // 遍历每个VLAN组
        for (let vlan_item of vlan_group) {
          // 如果VLAN组中包含连续的VLAN
          if (vlan_item.includes("-")) {
            // 将连续的VLAN拆分为开始VLAN和结束VLAN
            const vlan_items = vlan_item.split("-");
            const start_vlan = parseInt(vlan_items[0]);
            const end_vlan = parseInt(vlan_items[1]);
            // 将连续的VLAN添加到VLAN资源池中
            for (let j = start_vlan; j <= end_vlan; j++) {
              vlan_pool.push(j);
            }
            continue;
          }
          // 如果VLAN组中只有一个VLAN
          vlan_pool.push(parseInt(vlan_item));
        }
    
        // 对VLAN资源池进行升序排序
        vlan_pool.sort((a, b) => a - b);
    
        // 如果申请的VLAN在VLAN资源池中
        if (vlan_pool.includes(dest_vlan)) {
          // 从VLAN资源池中移除申请的VLAN
          vlan_pool.splice(vlan_pool.indexOf(dest_vlan), 1);
        }
    
        // 定义存储结果的列表
        const result = [];
        // 定义上一个VLAN的变量
        let last_vlan = null;
    
        // 遍历VLAN资源池中的每个VLAN
        for (let index = 0; index < vlan_pool.length; index++) {
          // 如果是第一个VLAN
          if (last_vlan === null) {
            result.push(vlan_pool[index].toString());
            last_vlan = vlan_pool[index];
            continue;
          }
          // 如果当前VLAN与上一个VLAN连续
          if (vlan_pool[index] - last_vlan === 1) {
            // 如果结果列表中的最后一个元素以"-上一个VLAN"结尾
            if (result[result.length - 1].endsWith("-" + last_vlan)) {
              // 将结果列表中的最后一个元素更新为"-当前VLAN"
              result[result.length - 1] = result[result.length - 1].slice(0, result[result.length - 1].lastIndexOf(last_vlan.toString())) + vlan_pool[index].toString();
            } else {
              // 在结果列表中添加"-当前VLAN"
              result.push("-" + vlan_pool[index].toString());
            }
          } else {
            // 在结果列表中添加",当前VLAN"
            result.push("," + vlan_pool[index].toString());
          }
          last_vlan = vlan_pool[index];
        }
    
        // 输出结果列表中的VLAN资源池
        console.log(result.join(""));
      });
    });
    
    
    ![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCodeMoreWhite.png)
## C++
    
    
    #include <iostream>
    #include <vector>
    #include <sstream>
    #include <algorithm>
    
    using  namespace std;
    int main() {
        string vlan_pool_input; // 存储输入的VLAN资源池的字符串
        getline(cin, vlan_pool_input); // 获取输入的VLAN资源池的字符串
        int dest_vlan; // 存储业务要申请的VLAN
        cin >> dest_vlan; // 获取业务要申请的VLAN
    
        vector<int> vlan_pool; // 存储VLAN资源池中的VLAN
        stringstream ss(vlan_pool_input); // 使用字符串流解析VLAN资源池的字符串
        string vlan_item; // 存储解析出的每个VLAN
        while (getline(ss, vlan_item, ',')) { // 按逗号分隔字符串，获取每个VLAN
            if (vlan_item.find('-') != string::npos) { // 如果VLAN是连续的范围
                stringstream range_ss(vlan_item); // 使用字符串流解析连续范围的字符串
                string start_vlan_str, end_vlan_str; // 存储连续范围的起始VLAN和结束VLAN
                getline(range_ss, start_vlan_str, '-'); // 获取起始VLAN
                getline(range_ss, end_vlan_str, '-'); // 获取结束VLAN
                int start_vlan = stoi(start_vlan_str); // 将起始VLAN转换为整数
                int end_vlan = stoi(end_vlan_str); // 将结束VLAN转换为整数
                for (int j = start_vlan; j <= end_vlan; j++) { // 将连续范围内的VLAN添加到VLAN资源池中
                    vlan_pool.push_back(j);
                }
            } else { // 如果VLAN是单个整数
                vlan_pool.push_back(stoi(vlan_item)); // 将VLAN转换为整数并添加到VLAN资源池中
            }
        }
    
        sort(vlan_pool.begin(), vlan_pool.end()); // 对VLAN资源池中的VLAN进行排序
    
        auto it = find(vlan_pool.begin(), vlan_pool.end(), dest_vlan); // 查找业务要申请的VLAN在VLAN资源池中的位置
        if (it != vlan_pool.end()) { // 如果找到了业务要申请的VLAN
            vlan_pool.erase(it); // 从VLAN资源池中移除业务要申请的VLAN
        }
    
        vector<string> result; // 存储最终输出结果的字符串向量
        int last_vlan = -1; // 存储上一个输出的VLAN
    
        for (int i = 0; i < vlan_pool.size(); i++) { // 遍历VLAN资源池中的VLAN
            if (last_vlan == -1) { // 如果是第一个输出的VLAN
                result.push_back(to_string(vlan_pool[i])); // 将VLAN转换为字符串并添加到结果向量中
                last_vlan = vlan_pool[i]; // 更新上一个输出的VLAN
                continue; // 继续下一次循环
            }
            if (vlan_pool[i] - last_vlan == 1) { // 如果当前VLAN与上一个输出的VLAN相差1
                if (result.back().find('-' + to_string(last_vlan)) != string::npos) { // 如果结果向量的最后一个字符串包含连续范围的结束VLAN
                    result.back() = result.back().substr(0, result.back().rfind(to_string(last_vlan))) + to_string(vlan_pool[i]); // 更新连续范围的结束VLAN
                } else { // 如果结果向量的最后一个字符串不包含连续范围的结束VLAN
                    result.push_back("-" + to_string(vlan_pool[i])); // 在结果向量中添加连续范围的结束VLAN
                }
            } else { // 如果当前VLAN与上一个输出的VLAN不相差1
                result.push_back("," + to_string(vlan_pool[i])); // 在结果向量中添加当前VLAN
            }
            last_vlan = vlan_pool[i]; // 更新上一个输出的VLAN
        }
    
        for (int i = 0; i < result.size(); i++) { // 遍历结果向量中的字符串
            cout << result[i]; // 输出结果向量中的字符串
        }
    
        return 0;
    }
    
    
    ![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCodeMoreWhite.png)
## C语言
    
    
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    
    // 定义最大VLAN池的大小
    #define MAX_VLAN 4096
    
    // 函数声明
    void parseAndFilterVlanPool(char *input, int *vlanPool, int *size, int destVlan);
    void formatVlanPool(int *vlanPool, int size, char *result);
    
    int main() {
        // 存储输入的VLAN资源池和业务申请的VLAN
        char input[1000];
        int destVlan;
    
        fgets(input, sizeof(input), stdin);
        input[strcspn(input, "\n")] = 0;  // 去除换行符
    
        scanf("%d", &destVlan);
    
        // 定义VLAN池数组，用于存储解析后的VLAN，最大为MAX_VLAN
        int vlanPool[MAX_VLAN];
        int size = 0;
    
        // 解析并过滤VLAN资源池（在解析过程中自动移除目标VLAN）
        parseAndFilterVlanPool(input, vlanPool, &size, destVlan);
    
        // 定义用于格式化输出的字符串
        char result[1000] = "";
        
        // 格式化VLAN资源池
        formatVlanPool(vlanPool, size, result);
        
        // 输出结果
        printf("%s\n", result);
    
        return 0;
    }
    
     
    void parseAndFilterVlanPool(char *input, int *vlanPool, int *size, int destVlan) {
        char *token = strtok(input, ",");  // 使用逗号分割字符串
        
        // 解析每个VLAN或VLAN范围
        while (token != NULL) {
            if (strchr(token, '-')) {
                // 如果包含连字符"-"，则为VLAN范围
                int start, end;
                sscanf(token, "%d-%d", &start, &end);  // 解析范围的起始和结束VLAN
                
                // 将范围内的VLAN逐个加入VLAN池，同时移除目标VLAN
                for (int i = start; i <= end; i++) {
                    if (i == destVlan) {
                        continue;  // 跳过目标VLAN
                    }
                    // 将VLAN加入有序数组
                    int j;
                    for (j = *size - 1; j >= 0 && vlanPool[j] > i; j--) {
                        vlanPool[j + 1] = vlanPool[j];
                    }
                    vlanPool[j + 1] = i;
                    (*size)++;
                }
            } else {
                // 如果是单个VLAN，直接处理
                int vlan = atoi(token);
                if (vlan == destVlan) {
                    token = strtok(NULL, ",");
                    continue;  // 跳过目标VLAN
                }
                // 将VLAN插入到有序数组中
                int j;
                for (j = *size - 1; j >= 0 && vlanPool[j] > vlan; j--) {
                    vlanPool[j + 1] = vlanPool[j];
                }
                vlanPool[j + 1] = vlan;
                (*size)++;
            }
            // 获取下一个VLAN或VLAN范围
            token = strtok(NULL, ",");
        }
    }
    
    // 格式化VLAN池为要求的字符串格式
    void formatVlanPool(int *vlanPool, int size, char *result) {
        int last = -1;  // 上一个处理的VLAN
        int start = -1; // 范围的起始VLAN
        
        for (int i = 0; i < size; i++) {
            if (last == -1) {
                // 第一个VLAN直接添加
                start = vlanPool[i];
                last = vlanPool[i];
            } else if (vlanPool[i] == last + 1) {
                // 如果当前VLAN与上一个连续，继续处理
                last = vlanPool[i];
            } else {
                // 如果不连续，检查是否是一个范围
                if (start == last) {
                    // 如果是单个VLAN，直接添加
                    char temp[20];
                    sprintf(temp, "%d,", start);
                    strcat(result, temp);
                } else {
                    // 否则为范围，添加"start-end"格式
                    char temp[40];
                    sprintf(temp, "%d-%d,", start, last);
                    strcat(result, temp);
                }
                // 处理下一个VLAN范围
                start = vlanPool[i];
                last = vlanPool[i];
            }
        }
        
        // 处理最后一个范围或VLAN
        if (start == last) {
            char temp[20];
            sprintf(temp, "%d", start);
            strcat(result, temp);
        } else {
            char temp[40];
            sprintf(temp, "%d-%d", start, last);
            strcat(result, temp);
        }
    
        // 去除最后的逗号
        if (result[strlen(result) - 1] == ',') {
            result[strlen(result) - 1] = '\0';
        }
    }
    
    
    ![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCodeMoreWhite.png)
