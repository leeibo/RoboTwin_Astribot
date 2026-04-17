之前我不小心让你回退了git版本，并且从历史文件中反编译去抢救一下VLM VQA的导出事宜，结果你修复后的VLM VQA导出完全不可用！回退git版本后到现在为止给你的指令在于/home/admin1/.codex/history_bak.jsonl，你可以查看后总结一下你应该怎么修复！具体是在执行ts为1776410337的命令后你的结果错了，我让你回滚，结果给我git全部回退导致改动全部丢失了。现在的功能和要求有偏差，要求完全重构现在的版本，因为你现在恢复后的版本完全不对不可用。当然历史的命令中有很多后面的指令是对前面的重构和优化，你需要从历史我的提问中提取出相关的有用的信息，少走弯路。

你至少要修复以下内容：

# VQA的链路修复：

对于已经获取的仿真数据中你要后处理生成每个episode对应的VQA，应该有以下几个部分：
- angle_delta：这部分VQA主要是推理出前后两个记忆帧之间的角度差值，这部分的两帧应该来自于同一个subtask的前两个stage，也就是不能跨物体操作。
- object_search：这是主要关注的VQA，是贯穿整个任务的主线。对于stage1和stage2，记录每段规划转角时的帧放进历史帧；对于action阶段每个chunk size记录一次记忆帧。
- memory_compression_vqa：每次切换子任务或者16帧将触发一次帧压缩，压缩的过程要保证：1.空间优先,也就是说保留下的记忆帧要尽可能覆盖更大的FOV视野范围 2.最新优先，也就是在满足1的情况下后进来的帧要尽可能去替换前面的老帧，比如在action阶段后进来的帧在压缩的时候肯定是只保留最后一帧。（实际上保证1的前提下尽可能保留新帧移除老帧，你可以参考记忆中的是怎么描述的）。另外对于压缩任务的生成，你需要再做以下几点优化\\改动:1.现在的VLM记忆压缩任务是到16帧才触发的,但是现在的VLM记忆压缩是一个贪心前缀和的过程:也就是说以往最优的帧+新帧的压\n  缩效果应该和以往所有帧+新帧的压缩结果一致.所以理论上来说可以构造非常多的压缩VQA,比如现在已有的压缩是将1-16帧压缩为了[1,16],那么实际上我们可以构造\n  新的VQA是从[1,3,7,16]\\[1,2,3,16]等等帧也压缩到[1,16],这样的话可以让我们的数据量快速scale up上去,并且没有定死必须16帧压缩的限制!所以理论上来说,你\n  可以在完成一次子任务或者到16帧的时候触发一次压缩,然后这次压缩你选取压缩后的最优帧和其余帧也可以构造出一堆的压缩数据,可以构造出从max(最优帧\n  数,4)~16帧压缩到最优帧的大量VQA! 



VQA的构造模版你可以参考：
在stage 1、2的时候的object_search：
user: <image>Your task is: "find the hammer with two-tone handle and grasp it" The input images are ordered from earliest to latest, and the last image is the current view. Please think about the next action and output it. Your response should be in the format of: <think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>.
assistant: <think>Frames: current only. Past actions: none. 目标: the hammer with two-tone handle。 当前视角中看到the hammer with two-tone handle，位置约在(499,613)。信息已足够，目标已在中心附近，开始执行动作。 Next: Rotate(0, 0).</think><info>1</info><frame>1</frame><camera>Rotate(0, 0)</camera><action><action_chunk></action>
user: <image>Your task is: "find green block and grasp it" The input images are ordered from earliest to latest, and the last image is the current view. Please think about the next action and output it. Your response should be in the format of: <think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>.
assistant: <think>Frames: current only. Past actions: none。 目标: green block。 当前视角中看到green block，位置约在(851,826)。信息已足够，向右转28度做居中。 Next: Rotate(-28, 0).</think><info>1</info><frame>1</frame><camera>Rotate(-28, 0)</camera><action></action>
如果是在历史记忆中发现的物体，要选用最新的已发现物体的那一个记忆帧，然后说当前视角中没看到xxx，在第k帧中看到目标物体，位置约在(xxx,yyy)。信息已足够，转xxx度，然后frame中填写k

在stage3的时候理论上和之前的Q保持一致，在think例如中可以写：
"value": "<think>Frames: current only. Past actions: none. The current task is “xxx”. Info sufficient. Next: Rotate(0, 0).</think><info>1</info><frame>[1]</frame><camera>Rotate(0, 0)</camera><action><action_chunk></action>

对于angle_delta，你可以参考：user: <image><image>给定同一子任务的历史帧与当前帧，任务是"find the toycar made from smooth plastic and pick it up"。请根据两张图回答：从历史帧到当前帧，机器人累计水平转了多少度？请输出 <think>...</think><answer>...</answer>。
assistant: <think>这两张图像来自同一个子任务，并且按时间顺序排列。从图像与中间记忆帧的视角变化看，中间累计的水平转动约为[(37, 0)]。因此从历史帧到当前帧，总共需要向左旋转37度。</think><answer>角度差值是37度，表示需要向左旋转。</answer>

对于帧筛选，你可以参考：
user: <image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>Your task is: "Build a concise visual memory for small block." The input images are ordered from earliest to latest, and the last image is the current view. Please remove redundant memory views while preserving the most relevant target evidence. Your response should be in the format of: <think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>.
assistant: <think>Frames: 16 total. View changes: [(-28, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (71, 0), (0, 0), (-1, 0), (0, 0), (0, 0), (0, 0), (-100, 0), (0, 0)]. To cover the newest useful scene as few frames as possible, I keep frames [1, 8, 14, 16]. Frames [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15] are redundant revisits or older weaker views of the same area, so I remove them. Info sufficient.</think><info>1</info><frame>[1, 8, 14, 16]</frame><camera></camera><action></action>

# randomized场景新增

你需要给当前的rotate任务也添加上扇形桌面的配置，注意新加入的散落物体要尽可能靠后，也就是处于r较大的区域，具体可以看我之前给你的记忆中总结出最新的要求。

未尽事宜请在我的历史指令中总结，忽略无关的指令