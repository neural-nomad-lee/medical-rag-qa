#!/usr/bin/env python3
# coding: utf-8
"""
generate_medical_classify_data.py —— 生成医疗版 BERT 查询分类训练数据

【用途】
  原训练集 model_generic_5000.json 全部为教育领域语料（课程/学费/大纲等），
  医疗问答场景下会把大量医疗问题误判为"通用知识"，绕过 RAG 检索链路。
  本脚本按模板组合生成"通用知识 / 专业咨询"两类均衡、多样的医疗版训练集。
  (数据量已按需求由 5000 条缩减为 300 条)

【输出】
  rag_qa/classify_data/medical_classify_300.json (JSONL, 每行: {"query": ..., "label": ...})

【用法】
  python scripts/generate_medical_classify_data.py [生成条数]   # 默认 300
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------- 医疗专业咨询语料（label=专业咨询） ----------------
MED_TOPICS = [
    "高血压", "低血压", "冠心病", "心律失常", "心力衰竭", "糖尿病", "甲亢", "甲减",
    "哮喘", "慢性支气管炎", "肺炎", "肺结节", "慢阻肺", "流感", "感冒", "咳嗽",
    "胃炎", "胃溃疡", "消化不良", "便秘", "腹泻", "脂肪肝", "胆囊炎", "痔疮",
    "肾炎", "尿路感染", "肾结石", "前列腺增生", "贫血", "白血病",
    "颈椎病", "腰椎间盘突出", "骨质疏松", "关节炎", "骨折", "扭伤",
    "偏头痛", "失眠", "焦虑症", "抑郁症", "面瘫", "脑梗", "癫痫",
    "湿疹", "荨麻疹", "痤疮", "银屑病", "带状疱疹", "手足癣",
    "近视", "干眼症", "白内障", "青光眼", "中耳炎", "耳鸣",
    "龋齿", "牙周炎", "口腔溃疡", "过敏性鼻炎", "鼻窦炎", "咽炎",
    "月经不调", "痛经", "多囊卵巢", "孕期补叶酸", "乳腺增生", "更年期综合征",
    "小儿发烧", "小儿咳嗽", "儿童疫苗接种", "新生儿黄疸", "佝偻病",
    "新冠疫苗", "体检报告", "血脂异常", "幽门螺杆菌", "甲状腺结节", "肿瘤筛查",
]

MED_TEMPLATES = [
    "{t}有哪些常见症状？",
    "{t}应该挂什么科？",
    "{t}怎么治疗？",
    "{t}日常生活中需要注意什么？",
    "{t}需要做哪些检查？",
    "{t}能根治吗？",
    "{t}饮食上有什么禁忌？",
    "{t}会遗传吗？",
    "{t}有什么并发症？",
    "{t}如何预防？",
    "得了{t}严重吗？",
    "{t}一般多久能好？",
    "怀疑自己有{t}应该怎么办？",
    "{t}患者可以运动吗？",
    "{t}看中医好还是西医好？",
    "{t}复发率高吗？",
    "帮我看看{t}的检查报告要注意什么",
    "{t}用药期间有什么注意事项？",
]

# ---------------- 通用知识/生活常识语料（label=通用知识） ----------------
GEN_TOPICS = [
    "Python", "Java", "人工智能", "机器学习", "区块链", "云计算", "大数据", "5G",
    "圆周率", "勾股定理", "一元二次方程", "微积分", "概率", "质数",
    "光合作用", "万有引力", "黑洞", "DNA", "板块运动", "臭氧层",
    "唐诗", "宋词", "红楼梦", "莎士比亚", "贝多芬", "莫扎特",
    "世界杯", "奥运会", "马拉松", "瑜伽", "健身",
    "咖啡", "茶叶", "巧克力", "红酒", "寿司",
    "旅行", "摄影", "画画", "书法", "围棋", "象棋",
    "太阳系", "月球", "彗星", "流星雨", "极光",
    "信用卡", "房贷", "基金", "理财", "保险",
    "垃圾分类", "低碳生活", "新能源汽车", "高铁", "飞机",
]

GEN_TEMPLATES = [
    "什么是{t}？",
    "{t}的基本原理是什么？",
    "帮我写一段关于{t}的代码",
    "解释一下{t}的概念。",
    "{t}是谁发明的？",
    "学习{t}有什么好方法？",
    "推荐几本{t}入门书籍",
    "{t}和{t2}有什么区别？",
    "写一首关于{t}的诗",
    "{t}的发展历史是怎样的？",
    "如何提高{t}的技能水平？",
    "{t}在生活中有哪些应用？",
    "怎么给{t}新手做入门计划？",
    "关于{t}有哪些有趣的冷知识？",
    "帮我列一个{t}学习清单",
]

GEN_FILLERS = [
    "计算1024乘以768等于多少？",
    "帮我写一个Python脚本读取CSV文件",
    "解释一下什么是RESTful API。",
    "今天天气怎么样？",
    "帮我生成一份周报模板",
    "把这段中文翻译成英文：今天是个好日子。",
    "如何培养孩子的时间管理能力？",
    "怎样养成规律作息的习惯？",
    "如何提高记忆力？",
    "写一个冒泡排序的示例代码",
    "SQL中的LEFT JOIN和INNER JOIN有什么区别？",
    "帮我起一个好听的网名",
    "怎么把PDF转换成Word？",
    "Excel如何做数据透视表？",
    "一句话概括《三体》讲了什么",
    "帮我写一封请假邮件",
    "怎样快速学会做番茄炒蛋？",
    "跑步之前需要做哪些热身动作？",
    "如何挑选一台性价比高的笔记本电脑？",
    "帮我制定一个一个月减脂计划",
]


def build_dataset(total: int = 300):
    random.seed(42)
    medical, general = set(), set()

    # 医疗专业咨询
    while len(medical) < total // 2:
        t = random.choice(MED_TOPICS)
        tpl = random.choice(MED_TEMPLATES)
        medical.add(tpl.format(t=t))
        if len(medical) >= total // 2:
            break

    # 通用知识: 模板组合 + 固定句子填充
    while len(general) < total - total // 2:
        if random.random() < 0.4:
            general.add(random.choice(GEN_FILLERS))
        else:
            t = random.choice(GEN_TOPICS)
            tpl = random.choice(GEN_TEMPLATES)
            general.add(tpl.format(t=t, t2=random.choice(GEN_TOPICS)))

    items = [{"query": q, "label": "专业咨询"} for q in medical] + \
            [{"query": q, "label": "通用知识"} for q in general]
    random.shuffle(items)
    return items


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out_path = os.path.join(ROOT, "rag_qa", "classify_data", f"medical_classify_{total}.json")
    items = build_dataset(total)

    # 简单质检: 标签均衡性
    n_med = sum(1 for i in items if i["label"] == "专业咨询")
    print(f"共生成 {len(items)} 条 | 专业咨询: {n_med} | 通用知识: {len(items) - n_med}")

    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
