"""最小测试：akshare 能否拉取 A 股数据"""
import os
import sys

# # 清空所有代理环境变量
# for key in list(os.environ.keys()):
#     if 'proxy' in key.lower():
#         del os.environ[key]

# # 然后设置 no_proxy 直连
# os.environ["NO_PROXY"] = "push2his.eastmoney.com"
# os.environ["no_proxy"] = "push2his.eastmoney.com"

import akshare as ak

try:
    df = ak.stock_zh_a_hist(
        symbol="300726",
        period="daily",
        start_date="20240101",
        end_date="20240110",
        adjust="qfq",
    )
    print(f"成功! {len(df)} 行")
    print(df.columns.tolist())
except Exception as e:
    print(f"失败: {e}")
