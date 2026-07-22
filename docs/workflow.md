# 协作流程

## 分支规范

main         稳定分支，14/14 通过，基线代码
feat/xxx     功能分支
perf/xxx     性能优化分支
fix/xxx      bug 修复
docs/xxx     文档

## 每日流程

### 开始工作
git checkout main && git pull && git checkout -b feat/xxx

### 改代码 -> 测试 -> 提交
pytest tests/ -v
python benchmarks/benchmark_prefill.py
git add -A && git commit -m "feat: xxx"
git push origin feat/xxx

### 开 PR
标题: [feat/perf/fix/docs] xxx
描述: 改动 + pytest 结果 + benchmark 数据

### 合并
git checkout main && git merge feat/xxx && git push origin main
git push origin --delete feat/xxx

## 底线规则
- 永不改 ref_rel_attn
- 改 kernel 必过 14 个测试
- 每次优化必贴 benchmark 数据
