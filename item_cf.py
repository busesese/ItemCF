# -*- coding: utf-8 -*-
# @Time    : 2020-12-18 16:34
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  ItemCF根据KDDCUP2020进行修改，通用型更强，带有测试样例供测试

from collections import defaultdict
import numpy as np
from tqdm import tqdm


def itemcf_sim(user_item_time_dict, location_weight=False, time_weight=False):
    """
    param: user_item_time_dict: dict 用户历史商品的点击及点击时间list eg:{user1:[(item1, time1), (item2, time2)]}, time1 and
    time2 must be string like '%Y-%m-%d' for days and '%Y-%m-%d %H:%M:%S' for second or int use directly
    param: location_weight: bool default false, 是否使用位置权重
    param: time_weight: bool default false,是否使用时间权重
    return Dict
    """
    # 入参检验
    if not isinstance(user_item_time_dict, dict):
        raise ValueError("input parameter user_item_time_dict must be a dict")
    
    item2item_sim = defaultdict(dict)
    item_cnt = defaultdict(int)
    
    # 计算两item相似性
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc1, (item1, click_time1) in enumerate(item_time_list):
            item_cnt[item1] += 1
            for loc2, (item2, click_time2) in enumerate(item_time_list):
                if item1 == item2:
                    continue
    
                # 位置权重
                if location_weight:
                    loc_alpha = 1 if loc1 > loc2 else 0.7
                    loc_weight = loc_alpha * (0.9 ** (np.abs(loc1 - loc2) - 1))
                else:
                    loc_weight = 1
                # 时间权重
                if time_weight:
                    if isinstance(click_time1, str):
                        import datetime
                        if len(click_time1) == 10:
                            time1 = datetime.datetime.strptime(click_time1, "%Y-%m-%d")
                            time2 = datetime.datetime.strptime(click_time2, "%Y-%m-%d")
                            click_time_weight = np.exp(0.7 ** np.abs((time1 - time2).days))
                        elif len(click_time1) == 19:
                            time1 = datetime.datetime.strptime(click_time1, "%Y-%m-%d %H:%M:%S")
                            time2 = datetime.datetime.strptime(click_time2, "%Y-%m-%d %H:%M:%S")
                            click_time_weight = np.exp(0.7 ** np.abs((time1 - time2).seconds))
                        else:
                            raise ValueError("if input time is string, must be like '%Y-%m-%d' for day and "
                                             "'%Y-%m-%d %H:%M:%S' for second")
                    elif isinstance(click_time1, int):
                        click_time_weight = np.exp(0.7 ** np.abs(click_time1 - click_time2))
                    else:
                        raise ValueError("input time type must be string or int")
                else:
                    click_time_weight = 1
                # 计算相似性
                item2item_sim[item1].setdefault(item2, 0)
                item2item_sim[item1][item2] += loc_weight * click_time_weight /np.log(len(item_time_list) + 1)
    
    i2i_sim = item2item_sim.copy()
    
    # 根据两item相似性进行热度降权
    for i, relate_items in i2i_sim.items():
        for j, wij in relate_items.items():
            i2i_sim[i][j] = wij / np.sqrt(item_cnt[i] * item_cnt[j])
    
    return i2i_sim


if __name__ == "__main__":
	# 各种格式的输入，主要区别在于时间的格式，目前支持三种，int类型，%Y-%m-%d时间格式， %Y-%m-%d %H:%M:%S时间格式
	d1 = {'w': [('1', '2020-10-23'), ('2', '2020-10-21'),  ('5', '2020-12-21')],
		'y': [('3', '2020-10-24'), ('1', '2020-10-25'),  ('4', '2020-11-21')],
		'd': [('1', '2020-10-13'), ('2', '2020-11-21'),  ('3', '2020-12-11')],
		's': [('1', '2020-11-23'), ('3', '2020-11-21'),  ('5', '2020-12-11')]}

	d2 = {'w': [('1', '2020-10-23 12:11:30'), ('2', '2020-10-21 08:11:30'),  ('5', '2020-12-21 20:04:31')],
		'y': [('3', '2020-10-24 12:11:30'), ('1', '2020-10-25 16:34:30'),  ('4', '2020-11-21 10:23:30')],
		'd': [('1', '2020-10-13 08:11:30'), ('2', '2020-11-21 17:55:21'),  ('3', '2020-12-11 03:04:30')],
		's': [('1', '2020-11-23 06:13:30'), ('3', '2020-11-21 09:23:37'),  ('5', '2020-12-11 17:45:30')]}

	d3 = {'w': [('1', 1), ('2', 7),  ('5', 12)],
		'y': [('3', 56), ('1', 23),  ('4', 15)],
		'd': [('1', 21), ('2', 7),  ('3', 43)],
		's': [('1', 27), ('3', 4),  ('5', 6)]}
	sim1 = itemcf_sim(d1, time_weight=True)
	print(sim1)
	sim2 = itemcf_sim(d2, time_weight=True)
	print(sim2)
	sim3 = itemcf_sim(d3, time_weight=True)
	print(sim3)
