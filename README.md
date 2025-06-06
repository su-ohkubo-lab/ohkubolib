# Ohkubo-Lab-Library (OhkuboLib)

## 目次
- [Ohkubo-Lab-Library (OhkuboLib)](#ohkubo-lab-library-ohkubolib)
  - [目次](#目次)
  - [想定環境](#想定環境)
  - [環境構築](#環境構築)
    - [1. Git をインストール (必須)](#1-git-をインストール-必須)
    - [2. Python の仮想環境を構築 (任意)](#2-python-の仮想環境を構築-任意)
    - [3. OhkuboLib を pip でインストール](#3-ohkubolib-を-pip-でインストール)
  - [ライブラリの構成](#ライブラリの構成)

## 想定環境
- MacOS
- Ubuntu

## 環境構築
### 1. Git をインストール (必須)
#### 以下のコマンドで Git がインストール済みかどうか確認できる. <!-- omit in toc -->
```sh
git --version

例)
    git version x.xx.x
```
例のように Git のバージョンが表示されたらインストール済みである.

Git をインストールしていない場合, [Git - Downloads](https://git-scm.com/downloads) を参照して Git をインストールする.

### 2. Python の仮想環境を構築 (任意)
```sh
python3 -m venv .venv
source .venv/bin/activate
```

#### 以下のコマンドで正しく仮想環境が構築できたか確認できる. <!-- omit in toc -->
```sh
which python3

例: カレントディレクトリが '/Users/xxx' の場合)
    /Users/xxx/.venv/bin/python3
```

### 3. OhkuboLib を pip でインストール
```sh
pip install --upgrade pip
pip install git+https://github.com/su-ohkubo-lab/ohkubolib.git
```

## ライブラリの構成
- [ohkubolib](#ohkubo-lab-library-ohkubolib)
  - [koopman](docs/koopman/README.md)