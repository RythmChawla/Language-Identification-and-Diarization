# import pandas as pd

# # Read the CSV file
# df1 = pd.read_csv('/home/teaching/Desktop/priyam/labels/vad_all_segments_all_audios.csv')  

# # Create the new 'label' column based on 'language_tag'
# df1['label'] = df1['language_tag'].apply(lambda x: 1 if x != 'NON_SPEECH' else 0)
# df1['start'] = df1['start']*1000
# df1['end'] = df1['end']*1000
# df1['length'] = df1['length']*1000

# #print(df1)
# # Save the modified DataFrame back to CSV
# df1.to_csv('/home/teaching/Desktop/priyam/labels/vadPred.csv', index=False)  # Save to a new file to preserve original
# ''''''''''''


# df2 = pd.read_csv('/home/teaching/Desktop/priyam/labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv')  # Replace with your file path

# # Initialize a list to store the new rows
# new_rows = []


# # Iterate through the DataFrame rows
# for i in range(len(df2) - 1):
#     current_row = df2.iloc[i]
#     next_row = df2.iloc[i + 1]
    
#     # Add the current row to the new list
#     new_rows.append(current_row.to_dict())
    
#     # Check if there's a gap between current_row['end'] and next_row['start']
#     if current_row['end'] != next_row['start']:
#         # Create a new row to fill the gap
#         new_row = {
#             'audio_name': current_row['audio_name'],
#             'utt_id': 'a',
#             'start': current_row['end'],
#             'end': next_row['start'],
#             'length': next_row['start'] - current_row['end'],
#             'language_tag': 'NON_SPEECH',
#             'overlap_diff_lang': False,
#             'dev_eval_status': 'dev'
#         }
#         new_rows.append(new_row)

# # Add the last row (since the loop stops at len(df)-1)
# new_rows.append(df2.iloc[-1].to_dict())

# # Convert the list of rows back to a DataFrame
# new_df = pd.DataFrame(new_rows)
# new_df['label'] = new_df['language_tag'].apply(
#     lambda x: 1 if x in ['English', 'Mandarin'] else 0
# )

# print(new_df)
# # Save the updated DataFrame to a new CSV file
# new_df.to_csv('/home/teaching/Desktop/priyam/labels/TrueLabelUp.csv', index=False)



# # def load_and_preprocess(file_path):
# #     df = pd.read_csv(file_path)
# #     # Convert to list of intervals with labels and track last end time
# #     intervals = []
# #     last_end = 0
# #     for _, row in df.iterrows():
# #         intervals.append((row['start'], row['end'], row['label']))
# #         if row['end'] > last_end:
# #             last_end = row['end']
# #     return intervals, last_end

# # def find_matching_times(file1_intervals, file2_intervals):
# #     non_speech_time = 0.0  # Both labels = 0
# #     language_time = 0.0    # Both labels = 1
# #     i = j = 0
    
# #     while i < len(file1_intervals) and j < len(file2_intervals):
# #         # Current intervals
# #         start1, end1, label1 = file1_intervals[i]
# #         start2, end2, label2 = file2_intervals[j]
        
# #         # Find overlapping interval
# #         overlap_start = max(start1, start2)
# #         overlap_end = min(end1, end2)
        
# #         if overlap_start < overlap_end:  # If there is an overlap
# #             if label1 == label2:
# #                 if label1 == 0:  # NON_SPEECH match
# #                     non_speech_time += overlap_end - overlap_start
# #                 else:            # Language match
# #                     language_time += overlap_end - overlap_start
            
# #             # Move the pointer that ends first
# #             if end1 < end2:
# #                 i += 1
# #             else:
# #                 j += 1
# #         else:
# #             # No overlap, move the pointer that starts first
# #             if start1 < start2:
# #                 i += 1
# #             else:
# #                 j += 1
                
# #     return non_speech_time, language_time

# # # Load files
# # file1 = "/home/teaching/Desktop/priyam/labels/TrueLabelUp.csv"

# # file2 = "/home/teaching/Desktop/priyam/labels/vadPred.csv"

# # # Process files and get durations
# # file1_intervals, file1_duration = load_and_preprocess(file1)
# # file2_intervals, file2_duration = load_and_preprocess(file2)

# # # Calculate matching times
# # non_speech_match, language_match = find_matching_times(file1_intervals, file2_intervals)

# # # Print results
# # print("=== File Durations ===")
# # print(f"File 1 total duration: {file1_duration:.2f} ms")
# # print(f"File 2 total duration: {file2_duration:.2f} ms")
# # print("\n=== Matching Durations ===")
# # print(f"NON_SPEECH matching time: {non_speech_match:.2f} ms")
# # print(f"Language matching time: {language_match:.2f} ms")
# # print(f"Combined matching time: {non_speech_match + language_match:.2f} ms")
# # print(f"\nMatching ratio (of shorter file): {100*(non_speech_match + language_match)/max(file1_duration, file2_duration):.1f}%")
# # print(f"\nLDER: {100-100*(non_speech_match + language_match)/max(file1_duration, file2_duration):.3f}%")


# # Load and preprocess grouped by audio_name
# def load_and_group(file_path):
#     df = pd.read_csv(file_path)
#     grouped = {}
#     for audio_name, group in df.groupby("audio_name"):
#         intervals = [(row["start"], row["end"], row["label"]) for _, row in group.iterrows()]
#         max_end = max(row["end"] for _, row in group.iterrows())
#         grouped[audio_name] = (intervals, max_end)
#     return grouped

# # Calculate matching durations per audio file
# def calculate_global_match(true_grouped, pred_grouped):
#     total_non_speech = 0
#     total_language = 0
#     total_duration = 0

#     for audio_name in true_grouped:
#         if audio_name not in pred_grouped:
#             print(f"⚠️ Skipping {audio_name}: not found in predictions.")
#             continue

#         true_intervals, true_duration = true_grouped[audio_name]
#         pred_intervals, pred_duration = pred_grouped[audio_name]
#         non_speech, language = find_matching_times(true_intervals, pred_intervals)
        
#         total_non_speech += non_speech
#         total_language += language
#         total_duration += max(true_duration, pred_duration)

#     return total_non_speech, total_language, total_duration

# # Load files grouped by audio_name
# true_grouped = load_and_group(file1)
# pred_grouped = load_and_group(file2)

# # Calculate total match
# non_speech_match, language_match, total_duration = calculate_global_match(true_grouped, pred_grouped)

# # Report
# print("=== Aggregated Results Across All Files ===")
# print(f"NON_SPEECH matching time: {non_speech_match:.2f} ms")
# print(f"Language matching time: {language_match:.2f} ms")
# print(f"Combined matching time: {non_speech_match + language_match:.2f} ms")
# print(f"\nMatching ratio (of total duration): {100*(non_speech_match + language_match)/total_duration:.1f}%")
# print(f"LDER: {100 - 100*(non_speech_match + language_match)/total_duration:.3f}%")


