# kabu-screener（日本株スイング候補サイト）

ユーザーはプログラミング未経験。依頼は普段の言葉で来るので、下の対応表で場所を特定すること。

## 作業ルール（トークン節約）
- ファイル全体を読まない。grep で該当関数・定数だけ探して最小限の行を変更する
- 変更後は main に commit & push し、何をどう変えたかを日本語3行以内で報告
- 判定ロジックを変えたら「Actionsで backtest を再実行してください」と一言添える

## ファイル
- screener.py：毎日の銘柄判定とサイト生成（docs/index.html, docs/council_input.json）
- backtest.py：過去検証（docs/backtest.html）。判定条件は screener.py の CHECKS を共有
- .github/workflows/daily.yml：平日16:30自動実行 / backtest.yml：手動実行のみ
- council_prompt.md：エージェント協議ルール（Claudeプロジェクト側に貼る用）

## 言葉 → 場所（screener.py）
- 売買代金の下限 → MIN_TURNOVER / 表示するRRの下限 → MIN_RR / 表示件数 → TOP_N / 協議に回す数 → COUNCIL_N
- ボックス下限・ボックス → box_bottom（幅 0.06〜0.20、下限から1.03以内、60日）
- ボックス上抜け・ブレイク → box_break（出来高2倍）
- 逆三尊 → inv_hs / 押し目・グランビル・ダウ・パーフェクトオーダー → pullback
- 売られすぎ・RSI・乖離 → oversold / 高値更新・新高値 → new_high
- 条件の追加・削除 → STRATEGIES と CHECKS の両方を変更
- 地合い → market_mood / サイトの見た目 → CSS, row_html, build_site
- 損切り・利確の決め方 → 各条件関数の sig(entry, target, stop, ...) の引数

## 言葉 → 場所（backtest.py）
- 検証銘柄数 → SAMPLE / 検証年数 → YEARS / 最大保有日数 → MAX_HOLD
