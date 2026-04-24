总体任务目标的变更
现有的扫描过程是不自然的，每个任务并不是说只有找到整个场景的物体才能开始操作，而是每次找到一个子任务对应的物体就可开始一定的action。
因此不能端到端训练整个任务，而是要对所有任务进行子任务的划分。具体来说，一个任务可以分为以下几个阶段：
1. 在正式执行前，一个任务会被分成多个subtask: [subtask1,subtask2,subtask3,...]
2. 具体来说，一个subtask包含了一次转动+一段action。
  比如说一个任务：把A放到B左边，那么一个完整的流程应该是两个子任务：1. {寻找A} [pick A] 2. {寻找B} [place A to B left]。我们不需要找到的A和B后再去执行[pick A] ，而是可以[pick A] 后带着A去找B。
3. 每次{寻找}过程又可分为两个stage：
  1. stage 1. 粗略寻找。如果初始时目标物体落在了视野中，则直接进入stage2；否则这个过程中每次规划都会以一个固定单位离散值的帧数倍进行旋转直到目标物体落在了视野中。比如以45°为单位，那么整个过程会是：如果初始的时候A没有落在视野中，那么先向左转45°，如果还没有则继续左转。。。如果在这个过程中某次A落在了视野中，则停止并且进入stage2。如果向左转已经看到了桌面边界还没有找到A（落入视野），那么将直接将下一步的目标规划到初始值的右侧45°并继续寻找，直到找到A进入stage2
  2. Stage 2. 精准定位。这个时候不再粗略寻找，而是可以一步将视野正对过去，此时转动的角度为一个精确值。毕竟已经出现在视野中可以直接正对过去了。
  3. 这样会涉及到一个问题：如果在前面的subtask执行的过程中，后面subtask的需要的物体如果出现在视野中过，那么就要将这个物体标记为已发现的物体。后面需要寻找这个物体的时候直接进入stage 2 即可。因此实际上可以将物体从左到右维护一个list，如果在执行任意stage1或者stage2的起始或者终结帧过程中有物体落在了视野中，那么将这个物体标记为已找到。如果在寻找一个物体的时候该物体已找到，那么直接进入stage2，否则进入stage1
4. 需要对所有的旋转任务进行子任务划分，每一个task应该包含：
  1. 原始的obs和原始的task instruction
  2. 每一帧应有状态标签，表示处于何等状态：stage1的1，stage2的2，以及执行action的3
  3. 每一帧应有subtask标签，表示处于何等子任务中如0,1,2...
  4. 子task instruction，需要以index进行区分，如0：ins0，1：ins1.  。。。实际上是每一帧subtask标签对应的instruction的map
5. 不仅要记录VLA的数据，也需要生成对应的VLM的数据来进行pretrain。比如把A放到B左边这么一个任务，可以拆分出来以下VLM的数据如：

<image>Your task is: \"{}\" The input images are ordered from earliest to latest, and the last image is the current view. Please think about the next action and output it. Your response should be in the format of: <think>...</think><info>...</info><frame>...</frame><camera>...</camera><action>...</action>.
  - <think>: 思路
  - <info>：信息是否完备，如果完备可以停止主动感知环节
  - <frame>：检索或者保留的帧的id号
  - <camera>：相机的动作
  - <action>：机械臂的动作
  
单物体寻找任务的多帧收集
  1. 输入 n 帧
  2. 输出推理文本+相机转动
    1. 推理内容包括：
      1. 总共有多少帧，有多少帧是历史帧
      2. 相邻观测帧之间的相机动作
      3. 当前阶段的动作原因（包括收集到了什么信息，还差哪些信息），以及下一步动作；若某个目标已在当前帧中，则补充其归一化像素位置
<think>The input contains a single frame, which is the current view. Because there is no history frame, I do not list any previous camera action. In the current view, I am in a station or transit area with multiple gates and a pillar in the center. There are a few signs visible, but none clearly indicate gate 3 or 4. The area is open with people moving in various directions. The task is to find a sign pointing to gate 3 and 4. Since there are no clear indicators in this view, rotating 180 degrees could reveal additional signs or directions on the opposite side that are not currently visible. Because the sign pointing to gate 3 and 4 is not visible yet, I need to keep turning in yaw until it comes into view. The next step is Rotate(180, 0).</think><info>0</info><frame>[]</frame><camera>Rotate(180, 0)</camera><action></action>


<think>Across the provided images, I see 1 historical frame plus the current frame, for 2 frames in total. From the earliest observation to the current frame, the camera actions are: [(-35, 20)]. The current view shows a station area with a sign overhead clearly indicating the direction to gates 3 and 4. The sign is centrally positioned, aligned with the task requirement to locate and center the sign. Since the sign is now in the center of the view, task information is enough. The sign pointing to gate 3 and 4 is already aligned in the current frame around normalized pixel (500, 500), so no further motion is needed. </think><info>1</info><frame>[]</frame><camera>Rotate(0, 0)</camera><action></action>
  
多物体寻找任务的多帧收集
  1. 输入 n 帧
  2. 输出推理文本+相机转动
    1. 推理内容包括：
      1. 总共有多少帧，有多少帧是历史帧
      2. 相邻观测帧之间的相机动作
      3. 当前阶段的动作原因（包括收集到了什么信息，还差哪些信息），以及下一步动作；若某个目标已在当前帧中，则补充其归一化像素位置
  
  
  需要考虑的点：
  1. 任务涉及多个物体，指令希望可以流畅自然，比如有生硬拼接的，有改写的
  2. 任务寻找物体的信息没有先后顺序，把A放到B上，先看到了B也合理
  3. 对于找到了某个物体，要接着找，过程中的think信息一直要说信息不够，还差什么，直到发出一个信息完全收集完毕的信号（<info>为1）