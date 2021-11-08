# allocation-time_select
行业景气度模型
1.	宏观因子对于行业的影响、宏观因子对于个股的影响
（1）行业内分散度；（2）宏观因子对行业的影响，对行业内个股分散情况的影响；（3）宏观影响个股大概可以归结为财务因子（杜邦等式）和产业链；（4）将这些因素和行业分散度联系起来
2.	风格因子对于行业的影响
（1）风格因子在每个行业内（相似行业内）的有效性；（2）根据有效性进一步分解行业子风格；（3）行业子风格的特征对应损失函数的定义（高内聚低耦合，主要是目标函数的确定，可以参照华泰）；（4）风格驱动还是行业驱动；（5）行业间聚类，并在每个子类中确定局部主导风格因素；（6）风格变化本身是如何产生的（宏观、情绪、业绩）
3.	行业内部扩散度
（1）行业指数的上涨或下跌是如何从牵头股票扩散开来的，有没有一些规律性的，如大概率哪几只股票会先领涨，中期是哪几只股票领涨，后期是哪几只股票领涨，分行业统计；（2）这些个股票都具有什么样的风格；（3）哪些股票涨的多，具有什么样的风格。
4.	判断行业热力图（位置）
（1）行业景气度持续性；（2）行业拥挤度；（3）资金面模型（北向资金模型、基金资金估计、龙虎榜资金模型）；（4）一致预期模型（调研信息模型、研报数据模型）；（5）增强模型（权重配置）。
景气度持续性模型构建，和行业拥挤度因子挖掘是整个模型的核心部分
此处是模型的核心部分，主要根据市场表现和双向情绪标记市场景气度状态，以及风险因子的挖掘来刻画市场的拥挤程度。
5.	模型的迭代更新
（1）风格的更新；（2）轮动周期，是否越来越快；（3）轮动速度加快会对上述模型的影响；（4）融合传统资产配置模型和行业轮动模型。
风控模型
1.	广发、华泰、海通等都采用了不同的因子进行构建。有别于基础Barra模型，确立以宏观因子、风格因子、基本面因子三类因子的风控模型。
2.	风格的确定：面向全市场；持续存在且可以作为股票收益的驱动力解释；容易找到代理变量，能够分析风格与标的收益风险等各项指标的关系，方便做因子择时；风格稳定。
3.	风控模型中加入一些市场新兴出现的风格，如抱团风格、北向资金风格。
4.	新风格挖掘的系统性方法：根据行业景气度模型将市场分成若干行业及子风格，遵循高内聚低耦合的原则，确立完成后进行聚类和统计学习方法，重新寻找新的归类方法和区分变量，区分变量即为风格。
基金评级
1.	判断基金风格，基准指数确定
2.	分离出超额收益，建立起超额收益评级模型
3.	因子有效性检验后，需要对因子加上权重，权重取因子拥挤度、因子景气度、因子ICIR等因素的综合，获取一个稳定的，有别于等权的权重。
4.	估算基金持仓，并在基金公布新一期持仓后，更新基金在持仓公布前的持仓估计，预估/检验更新基金的重要调仓时点。
5.	构建超额收益的绩效归因模型。
6.	找出基金的选股择时能力：（1）通过改良版TM模型，重新区分基金的选股与择时收益，并获取影响基金收益的重要的alpha和择时因素；（2）穿透底层持仓，比较基金收益主要由哪些持仓个股的收益带来的；（3）对动态持仓进行蒙特卡洛模拟——两类蒙特卡洛，选取任意子组合，选取任意时段，找到基金收益贡献的重要个股因素。
本质上，前几个模块都是为了逐渐接近基金或组合的操作水平的问题。收益率是表象，基于持仓的选股或择时策略分析才是本质，目前的目标是清晰构建出基金经理的决策历史，历史上做出的正确决策越多，基金经理的能力越强。做成结构化数据，简化上层评级的判断模型。
7.	基金评级考虑基金自身规模因素、基金运营限制因素，在经理评级的时候剔除掉
8.	构建标签体系：分为底层基础指标，中层衍生指标，上层画像指标三类。以收益与风险数据及分布、相关收益风险指标，持仓行业与风格、重仓股表现、收益归因和Barra因子收益分解、风险暴露等基础数据；以底层持仓行为金融风格等特色衍生指标（如抱团风格、北向资金风格、涨幅扩散度风格、行业内子风格）；以基金评级、基金持仓偏好、基金持仓收益效果等上层风格。
基金资产配置模型
1.	宏观、中观、微观因子配比；
2.	各行业配置方案的不同；
3.	基础资产配置组合，这个在控制好风险的情况下可以用传统资产配置模型结果；
4.	衍生资产配置组合：景气度模型、有效因子挑选组合、扩散度模型等，用偏离度模型融合基础资产配置组合和衍生资产配置组合；
5.	回测、跟踪，迭代模型。
![image](https://user-images.githubusercontent.com/51026474/140695792-3206f07c-e19b-461d-8195-e375e909cfad.png)
