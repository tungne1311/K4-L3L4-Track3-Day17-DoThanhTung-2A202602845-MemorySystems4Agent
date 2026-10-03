# Báo cáo Day 17: Memory Systems for AI Agent

## Tóm tắt

Em xây hai agent và chạy chúng trên cùng một bộ dữ liệu. Baseline chỉ nhớ trong một thread. Advanced có thêm `User.md` để nhớ lâu dài, và compact memory để nén hội thoại dài. Kết quả gói gọn trong ba ý:

- Advanced nhớ được thông tin người dùng qua phiên mới (recall 100% so với 0%), và luôn giữ thông tin mới nhất khi người dùng đính chính.
- Ở hội thoại ngắn, Advanced **tốn hơn** Baseline 79% prompt token, vì mỗi lượt phải mang theo `User.md` và system prompt dài hơn.
- Ở hội thoại rất dài, Advanced **tiết kiệm** 50% prompt token, và phần tiết kiệm này đến hoàn toàn từ compact memory.

Toàn bộ số liệu dưới đây lấy từ chế độ offline (chạy xác định, lần nào cũng ra kết quả giống nhau, không cần API key), với ngưỡng compact 500 token và giữ lại 4 tin nhắn gần nhất.

## Hệ thống hoạt động thế nào

Mỗi lượt chat, Advanced Agent đi qua ba lớp bộ nhớ:

1. **Short-term**: các tin nhắn gần đây của thread được giữ nguyên văn.
2. **Persistent (`User.md`)**: những thông tin ổn định về người dùng (tên, nơi ở, nghề, style trả lời, sở thích, món ăn, thú cưng) được trích ra từ tin nhắn và ghi vào file. File này tồn tại qua mọi thread, kể cả khi khởi động lại agent.
3. **Compact**: khi các tin nhắn trong thread vượt 500 token, phần cũ được gộp thành một bản tóm tắt tối đa 10 dòng, chỉ giữ 4 tin nhắn gần nhất.

Prompt của mỗi lượt gồm: system prompt + phần Facts của `User.md` + bản tóm tắt + các tin nhắn gần đây. Phần Changelog trong `User.md` chỉ để tra cứu lịch sử thay đổi, không đưa vào prompt.

Baseline thì đơn giản hơn nhiều: mỗi lượt gửi lại toàn bộ lịch sử của thread. Sang thread mới, nó bắt đầu lại từ con số không.

## Kết quả benchmark

Dòng **Advanced (no compact)** là thí nghiệm đối chứng (ablation): vẫn là Advanced nhưng tắt compact. Nhờ dòng này mới tách được đâu là tác dụng của `User.md`, đâu là tác dụng của compact.

**Standard Benchmark** (10 hội thoại ngắn, khoảng 10 lượt mỗi hội thoại)

| Agent | Agent tokens | Prompt tokens | Recall | Chất lượng | Memory growth | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 3,120 | 15,584 | 0% | 0.20 | 0 B | 0 |
| Advanced | 2,844 | 27,866 | 100% | 1.00 | 777 B | 0 |
| Advanced (no compact) | 2,844 | 27,866 | 100% | 1.00 | 777 B | 0 |

**Long-Context Stress Benchmark** (1 hội thoại 16 lượt, mỗi lượt khoảng 600 ký tự)

| Agent | Agent tokens | Prompt tokens | Recall | Chất lượng | Memory growth | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2,731 | 22,463 | 0% | 0.20 | 0 B | 0 |
| Advanced | 2,737 | 11,262 | 100% | 1.00 | 450 B | 8 |
| Advanced (no compact) | 2,737 | 24,154 | 100% | 1.00 | 450 B | 0 |

**Prompt token của một lượt** tại một số thời điểm trong stress test:

| Lượt | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| Baseline | 205 | 684 | 1,344 | 1,951 | 2,549 |
| Advanced | 278 | 600 | 766 | 883 | 845 |

## Phân tích

### Vì sao Advanced nhớ tốt hơn

Câu hỏi recall luôn được hỏi ở **thread mới**. Với Baseline, thread mới nghĩa là trống trơn: nó trả lời "mình chưa có thông tin" cho mọi câu nên recall bằng 0. Điều này đúng như thiết kế, vì Baseline vẫn nhớ bình thường trong cùng một thread, và test `test_cross_session_recall` kiểm tra cả hai chiều.

Advanced đọc `User.md` nên trả lời được ngay. Khó hơn là trả lời **đúng**, vì dữ liệu cố tình cài bẫy:

- Nơi ở đổi từ Đà Nẵng sang Huế (bộ standard), và từ Huế sang Đà Nẵng (bộ stress).
- Nghề đổi từ backend engineer sang MLOps engineer.
- "Product manager" chỉ là câu đùa, và "Hà Nội" chỉ là nơi đi họp.

Agent vượt qua các bẫy này nhờ ba quy tắc khi trích xuất: bỏ qua mệnh đề phủ định ("không còn ở…", "chỉ là nơi…"), bỏ qua câu đùa, và không bao giờ lấy thông tin từ câu hỏi. Khi người dùng đính chính, giá trị mới ghi đè giá trị cũ, nên câu trả lời không bao giờ chứa cùng lúc "Huế" lẫn "Đà Nẵng".

### Vì sao Advanced tốn hơn ở hội thoại ngắn

Ở bộ standard, mỗi hội thoại chỉ khoảng 250 token, chưa lần nào chạm ngưỡng compact (0 compactions). Advanced vì thế phải gánh chi phí cố định mà không được bù lại gì:

- System prompt dài hơn khoảng 43 token, vì có thêm hướng dẫn dùng bộ nhớ.
- Phần Facts của `User.md` tăng dần đến khoảng 88 token vào cuối benchmark.

Nhân với 115 lượt (gồm cả câu hỏi recall), phần chi phí này vào khoảng 12 nghìn token, đúng bằng khoảng chênh giữa 27,866 và 15,584. Hai dòng Advanced và Advanced (no compact) giống hệt nhau, khẳng định compact không đóng vai trò gì ở hội thoại ngắn.

Nói gọn: **ở hội thoại ngắn, ta trả thêm token để mua khả năng nhớ qua phiên.** Nếu sản phẩm chủ yếu có các phiên ngắn mà người dùng không quay lại, cái giá đó có thể không đáng.

### Vì sao compact giúp Advanced có lợi thế ở hội thoại dài

Baseline gửi lại toàn bộ lịch sử ở mỗi lượt, nên prompt của một lượt tăng tuyến tính: từ 205 token ở lượt 1 lên 2,549 token ở lượt 16. Tổng chi phí vì thế tăng theo bình phương số lượt.

Advanced dao động quanh 750 đến 900 token sau lượt 8, vì mỗi khi vượt ngưỡng, phần cũ bị nén lại. Tổng cộng, Advanced tốn 11,262 token so với 22,463 của Baseline, tức giảm 50%.

Dòng ablation chứng minh phần tiết kiệm này đến từ compact chứ không phải từ `User.md`. Tắt compact đi, Advanced tốn 24,154 token, còn nhiều hơn cả Baseline, vì vẫn mang toàn bộ lịch sử cộng thêm `User.md`.

Cũng cần nói rõ: compact chỉ giảm **prompt tokens processed**, tức lượng ngữ cảnh phải đưa lại vào model ở mỗi lượt. Nó không làm câu trả lời ngắn đi, nên cột Agent tokens gần như không đổi (2,737 so với 2,731).

Ngưỡng compact là một nút vặn đánh đổi:

| Ngưỡng | Prompt tokens (stress) | So với Baseline | Compactions | Compactions (standard) |
|---|---|---|---|---|
| 250 | 10,049 | −55% | 28 | 3 |
| 500 (mặc định) | 11,262 | −50% | 8 | 0 |
| 1000 | 14,309 | −36% | 2 | 0 |
| 2000 | 18,318 | −18% | 1 | 0 |

Ngưỡng càng thấp thì càng tiết kiệm, nhưng nén càng nhiều lần. Ở chế độ live, mỗi lần nén là một lần gọi LLM để tóm tắt (vừa tốn tiền vừa chậm), và mỗi lần nén lại có thêm nguy cơ mất chi tiết. Ở ngưỡng 250, ngay cả hội thoại ngắn của bộ standard cũng bị nén 3 lần mà gần như không tiết kiệm được gì. Em chọn 500 vì đó là mức cân bằng với bộ dữ liệu này.

### Bản tóm tắt có làm mất thông tin không

Có, và chính lúc làm bài em đã gặp. Phiên bản đầu tiên lấy câu đầu của mỗi tin nhắn. Kết quả là bản tóm tắt toàn những câu như "Đã ghi nhận." hay "Mình tăng độ khó thêm một chút…", còn bốn chủ đề tin tức quan trọng (Artemis III, X-59, El Nino, kế hoạch điện của British Columbia) thì mất sạch.

Bản sửa làm ba việc:

- Bỏ các câu xác nhận rỗng của assistant.
- Bỏ các câu trả lời recall của assistant, vì chúng chỉ lặp lại `User.md` vốn đã có trong prompt.
- Với mỗi tin nhắn của người dùng, chọn câu nhiều thông tin nhất (nhiều tên riêng, con số) thay vì câu đầu.

Bản tóm tắt sau khi sửa giữ được các chủ đề tin tức và câu đính chính nơi ở. Đổi lại, prompt tăng nhẹ: mức tiết kiệm giảm từ 54% xuống 50%. Em chấp nhận đánh đổi này.

Rủi ro vẫn còn. Bản tóm tắt bị giới hạn 10 dòng và bỏ dòng cũ nhất khi đầy, nên trong hội thoại rất dài, những chủ đề nhắc từ đầu (như tin Artemis III ở lượt 1) cuối cùng vẫn bị đẩy ra ngoài. Điều cứu cho recall là **các thông tin quan trọng về người dùng không nằm trong bản tóm tắt mà nằm trong `User.md`**. Vì vậy dù tóm tắt có bỏ sót, agent vẫn nhớ tên, nghề, nơi ở. Đây là lý do nên tách rõ hai lớp: thông tin ổn định thì lưu vào file, còn diễn biến hội thoại thì tóm tắt.

### `User.md` tăng trưởng ra sao và có rủi ro gì

Sau 10 phiên, `User.md` của bộ standard nặng 777 byte (8 thông tin và 9 dòng changelog). Nghe thì nhỏ, nhưng có hai rủi ro thật:

1. **File phình theo thời gian.** Mỗi byte trong phần Facts bị đưa vào prompt ở mọi lượt, nên chi phí của file nhân lên theo số lượt chat. Em chặn bằng các giới hạn cứng: style tối đa 8 mục, interests tối đa 6 mục, changelog tối đa 20 dòng, và changelog không được đưa vào prompt. Thiết kế ban đầu ghi lại toàn bộ danh sách style cũ và mới mỗi lần thêm một mục, làm file lên tới 1,818 byte. Sau khi đổi sang chỉ ghi phần thay đổi (`style: +nhấn trade-off`), file còn 777 byte.
2. **Lưu sai thông tin.** Một thông tin sai trong `User.md` còn nguy hiểm hơn việc không nhớ, vì nó lặp lại ở mọi phiên sau. Phần bonus bên dưới chủ yếu nhắm vào rủi ro này.

## Phần bonus

### 1. Confidence threshold

**Giải quyết vấn đề gì:** người dùng hay nói những câu chưa chắc chắn, kiểu "Có lẽ mình sẽ chuyển ra Hà Nội". Nếu lưu ngay, agent sẽ coi một dự định là sự thật.

**Cách làm:** mỗi thông tin trích ra có điểm tin cậy. Mức gốc là 0.7, cộng 0.2 nếu câu có dấu hiệu khẳng định hoặc đính chính ("giờ", "thực ra", "hiện tại"), trừ 0.5 nếu câu có dấu hiệu phỏng đoán ("có lẽ", "hình như", "dự định"). Chỉ thông tin đạt từ 0.5 trở lên mới được ghi vào file. Câu ví dụ ở trên chỉ đạt 0.2 nên bị bỏ, còn "Mình chuyển ra Hà Nội rồi" thì được lưu.

**Cải thiện gì:** giữ cho recall đúng, vì tránh được việc lưu sai nơi ở. Về token thì ảnh hưởng không đáng kể.

**Rủi ro:** danh sách từ khóa không thể đầy đủ. Một câu chắc chắn nhưng diễn đạt lạ có thể bị bỏ sót, và ngưỡng 0.5 là chọn bằng cảm tính chứ chưa được hiệu chỉnh trên dữ liệu thật.

### 2. Conflict handling (xử lý đính chính)

**Giải quyết vấn đề gì:** khi người dùng đính chính, agent không được giữ cùng lúc thông tin cũ (đã sai) lẫn thông tin mới.

**Cách làm:** thông tin dạng một giá trị (nơi ở, nghề) thì bị ghi đè, và giá trị cũ được chuyển sang Changelog (`location: Đà Nẵng -> Huế`). Thông tin dạng danh sách (style, sở thích) thì được gộp lại. Riêng style có quy tắc thay thế: "3 bullet" thay cho "dạng bullet". Trong cùng một tin nhắn, mệnh đề phủ định bị bỏ qua, nên câu "giờ ở Huế chứ không còn ở Đà Nẵng" chỉ lưu Huế.

**Cải thiện gì:** đây là lý do câu hỏi "Nếu ai đó nhắc Huế, Hà Nội hay product manager…" đạt điểm tối đa. Test `test_correction_keeps_latest_fact_only` khẳng định câu trả lời không còn chứa "backend" hay "Đà Nẵng".

**Rủi ro:** quy tắc "mới nhất thắng" có thể sai. Nếu người dùng nhắc lại thông tin cũ trong một câu bình thường không có từ phủ định, agent sẽ cập nhật ngược lại. Changelog giúp phát hiện và hoàn tác, nhưng hiện chưa có cơ chế tự hoàn tác.

### 3. Entity extraction có cấu trúc

**Giải quyết vấn đề gì:** nếu ghi nguyên văn câu chat vào `User.md`, file vừa dài vừa khó cập nhật.

**Cách làm:** thông tin được tách thành các field cố định (name, location, profession, style, interests, favorite_drink, favorite_food, pet). Mỗi field nằm trên một dòng `- key: value`, nên vừa dễ cập nhật đúng chỗ vừa dễ đọc.

**Cải thiện gì:** phần Facts chỉ khoảng 88 token sau 10 phiên, nhờ đó chi phí cố định ở mỗi lượt được giữ thấp. Recall cũng chính xác vì câu trả lời lấy đúng field cần thiết.

**Rủi ro:** bộ trích xuất dùng regex cho tiếng Việt nên khá giòn. Nó chỉ hiểu các cách diễn đạt đã gặp trong dữ liệu, và thông tin nào nằm ngoài danh sách field sẽ bị bỏ qua. Trong sản phẩm thật nên thay bằng LLM trích xuất có schema, rồi vẫn giữ confidence threshold ở phía sau.

## Giới hạn của kết quả

- **Recall 100% có phần "dễ" do chế độ offline.** Câu trả lời được ghép thẳng từ `User.md`, nên con số này đo độ chính xác của phần trích xuất và lưu trữ, chưa phải chất lượng trả lời của một LLM thật. Điểm chất lượng cũng là heuristic: đủ thông tin, ngắn gọn, có bullet.
- **Token là ước lượng** (khoảng 4 ký tự một token), không phải tokenizer thật. Tỷ lệ giữa các agent vẫn đáng tin, nhưng con số tuyệt đối thì không.
- **Chế độ live đã được dựng nhưng chưa chạy với API thật.** Đã dựng gồm LangChain `create_agent`, `InMemorySaver`, tool đọc và ghi `User.md`, `SummarizationMiddleware`. Khi chạy live, mỗi lần compact sẽ phát sinh thêm một lần gọi LLM để tóm tắt, chi phí mà bảng offline không thể hiện.
- Chưa làm **memory decay** (giảm ưu tiên thông tin cũ theo thời gian). Hiện chỉ có giới hạn số mục, theo kiểu giữ mục mới nhất.

## Kết luận

Câu chuyện xuyên suốt khớp với kỳ vọng của bài:

1. Baseline không nhớ dài hạn.
2. `User.md` giúp recall tăng từ 0% lên 100%.
3. Hội thoại dài làm chi phí prompt của Baseline tăng nhanh.
4. Compact kéo chi phí đó xuống một nửa.

Cái giá phải trả là hệ thống phức tạp hơn. Có thêm ngưỡng phải chỉnh, bản tóm tắt có thể làm rơi thông tin, và một file bộ nhớ có thể lưu sai. Vì vậy phần đáng đầu tư nhất không phải là "nhớ nhiều hơn", mà là các cơ chế bảo vệ quyết định **cái gì được phép nhớ**.

## Chạy lại

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest

python src/benchmark.py           # in 2 bảng + ablation + prompt theo từng lượt
pytest src/test_agents.py -v      # 7 test
COMPACT_THRESHOLD_TOKENS=1000 python src/benchmark.py   # thử ngưỡng khác
python src/benchmark.py --live    # cần .env: LLM_PROVIDER, LLM_MODEL, API key
```
