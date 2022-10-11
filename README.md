# Multimodal RSSM

Implementation of Multimodal Recurrent State Space Model (MRSSM).

## Abstract of Multimodal RSSM
...

## Execution environment
dockerフォルダ下のDockerfileを使用してdocker環境を構築して使用してください．

シミュレーション環境を使用する場合は，with_mujocoフォルダ下のDockerfileを使用してください．

## Execution procedure
MRSSMの学習
1. デモンストレーションデータをdataset/TASK_NAMEフォルダ下へ保存
2. train/ENV_NAME/TASK_NAME/MRSSM/MRSSM フォルダへ移動
3. config/train.yamlを調整
4. python main.py main.experiment_name="sample"

## Notes  
...

Original paper:
...

