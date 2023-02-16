mkdir dump/raw/valid-reminder-25spis
cp dump/raw/valid/feats_type dump/raw/valid-reminder-25spis/.
grep "reminder_eval_" dump/raw/valid/text > dump/raw/valid-reminder-25spis/text
grep "reminder_eval_" dump/raw/valid/wav.scp > dump/raw/valid-reminder-25spis/wav.scp

mkdir dump/raw/valid-weather-25spis
cp dump/raw/valid/feats_type dump/raw/valid-weather-25spis/.
grep "weather_eval_" dump/raw/valid/text > dump/raw/valid-weather-25spis/text
grep "weather_eval_" dump/raw/valid/wav.scp > dump/raw/valid-weather-25spis/wav.scp

mkdir dump/raw/test-reminder
cp dump/raw/test/feats_type dump/raw/test-reminder/.
grep "reminder_test_" dump/raw/test/text > dump/raw/test-reminder/text
grep "reminder_test_" dump/raw/test/wav.scp > dump/raw/test-reminder/wav.scp

mkdir dump/raw/test-weather
cp dump/raw/test/feats_type dump/raw/test-weather/.
grep "weather_test_" dump/raw/test/text > dump/raw/test-weather/text
grep "weather_test_" dump/raw/test/wav.scp > dump/raw/test-weather/wav.scp
