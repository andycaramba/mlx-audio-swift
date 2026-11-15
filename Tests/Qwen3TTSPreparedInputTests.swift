import Foundation
@testable import MLXAudioTTS
import XCTest

final class Qwen3TTSPreparedInputTests: XCTestCase {
    func testValidPreparedInputReachesMaterializerOnceWithUnchangedBytesAndEmbeddedConfig() throws {
        let files = validFiles()
        var calls = 0
        var result: String?
        XCTAssertNoThrow(result = try Qwen3TTSPreparedInput.withPreparedInput(files) { input in
            calls += 1
            XCTAssertEqual(input.modelWeights, Data([0x11, 0x22]))
            XCTAssertEqual(input.speechWeights, Data([0x33, 0x44]))
            XCTAssertEqual(input.config.ttsModelType, "custom_voice")
            XCTAssertEqual(input.config.tokenizerConfig?.decodeUpsampleRate, 123)
            XCTAssertEqual(input.speechConfig.decodeUpsampleRate, 1921)
            XCTAssertEqual(input.tokenizer.encode(text: "a"), [1])
            XCTAssertEqual(input.tokenizer.encode(text: "ab"), [3])
            return "prepared"
        })
        XCTAssertEqual(result, "prepared")
        XCTAssertEqual(calls, 1)
    }

    func testEveryConsumedFileIsRequiredBeforeMaterialization() throws {
        for name in ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors",
                     "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors"] {
            var files = validFiles()
            files.removeValue(forKey: name)
            assertRejected(files, expected: .missingFile(name))
        }
    }

    func testPreparedTokenizerPreservesBinaryDistinctUnicodeVocabulary() throws {
        var files = validFiles()
        files["tokenizer.json"] = Data(#"{"model":{"type":"BPE","vocab":{"é":4,"e\u0301":5},"merges":[]}}"#.utf8)
        var calls = 0
        try Qwen3TTSPreparedInput.withPreparedInput(files) { input in
            calls += 1
            XCTAssertEqual(input.tokenizer.convertTokenToId("\u{00E9}"), 4)
            XCTAssertEqual(input.tokenizer.convertTokenToId("e\u{0301}"), 5)
        }
        XCTAssertEqual(calls, 1)
    }

    func testPreparedTokenizerPreservesScalarTypesAndNestedConfig() throws {
        for (scalar, expected) in [("1", [1]), ("1.5", [1, 1]), ("true", [1, 1]),
                                   ("null", [1, 1]), (#""1""#, [1, 1]),
                                   ("18446744073709551615", [1, 1])] {
            var files = validFiles()
            files["tokenizer_config.json"] = Data("""
                {"tokenizer_class":"Qwen2Tokenizer","model_max_length":\(scalar),
                 "chat_template":[{"name":"default","template":"aa"}]}
                """.utf8)
            var calls = 0
            try Qwen3TTSPreparedInput.withPreparedInput(files) { input in
                calls += 1
                XCTAssertEqual(try input.tokenizer.applyChatTemplate(messages: [], truncation: true), expected, scalar)
            }
            XCTAssertEqual(calls, 1, scalar)
        }
    }

    func testMalformedModelConfigStopsBeforeMaterialization() {
        var files = validFiles()
        files["config.json"] = Data("{".utf8)
        assertRejected(files, expected: .invalidModelConfiguration)
    }

    func testOnlyExplicitCustomVoiceConfigReachesMaterialization() {
        for config in [
            #"{"model_type":"qwen3_tts","tts_model_type":"base","talker_config":{}}"#,
            #"{"model_type":"qwen3_tts","tts_model_type":"voice_design","talker_config":{}}"#,
            #"{"model_type":"other","tts_model_type":"custom_voice","talker_config":{}}"#,
            #"{"tts_model_type":"custom_voice","talker_config":{}}"#,
            #"{"model_type":"qwen3_tts","talker_config":{}}"#,
            #"{"model_type":"qwen3_tts","tts_model_type":"custom_voice"}"#
        ] {
            var files = validFiles()
            files["config.json"] = Data(config.utf8)
            assertRejected(files, expected: .invalidModelConfiguration)
        }
    }

    func testMalformedOrMissingSpeechDecoderStopsBeforeMaterialization() {
        for config in ["{", "{}", #"{"decoder_config":"invalid"}"#] {
            var files = validFiles()
            files["speech_tokenizer/config.json"] = Data(config.utf8)
            assertRejected(files, expected: .invalidSpeechConfiguration)
        }
    }

    func testMalformedTokenizerJSONStopsBeforeMaterialization() {
        for name in ["tokenizer_config.json", "tokenizer.json"] {
            var files = validFiles()
            files[name] = Data("{".utf8)
            assertRejected(files, expected: .invalidTokenizer)
        }
    }

    func testUnpreparedTokenizerStopsBeforeMaterializationWithoutRepair() {
        for tokenizer in [
            #"{}"#,
            #"{"model":{"vocab":{"a":1}}}"#,
            #"{"model":{"merges":[],"vocab":null}}"#,
            #"{"model":{"merges":["broken"],"vocab":{"a":1}}}"#,
            #"{"model":{"merges":[["a"]],"vocab":{"a":1}}}"#,
            #"{"model":{"merges":[["a",1]],"vocab":{"a":1}}}"#,
            #"{"model":{"merges":[1],"vocab":{"a":1}}}"#
        ] {
            var files = validFiles()
            files["tokenizer.json"] = Data(tokenizer.utf8)
            assertRejected(files, expected: .invalidTokenizer)
        }
    }

    func testUnsupportedTokenizerClassStopsBeforeMaterialization() {
        for config in [#"{"tokenizer_class":"UnsupportedTokenizer"}"#, "{}"] {
            var files = validFiles()
            files["tokenizer_config.json"] = Data(config.utf8)
            assertRejected(files, expected: .invalidTokenizer)
        }
    }

    func testOtherAdmittedFilesAreIgnoredAndDoNotReplaceConsumedFiles() throws {
        var files = validFiles()
        files["unused.json"] = Data("{".utf8)
        files["other.safetensors"] = Data([0xFF])
        var calls = 0
        try Qwen3TTSPreparedInput.withPreparedInput(files) { input in
            calls += 1
            XCTAssertEqual(input.modelWeights, Data([0x11, 0x22]))
        }
        XCTAssertEqual(calls, 1)
        files.removeValue(forKey: "model.safetensors")
        assertRejected(files, expected: .missingFile("model.safetensors"))
    }

    func testMaterializerFailurePropagatesWithoutRetry() {
        enum Failure: Error { case materializer }
        var calls = 0
        XCTAssertThrowsError(try Qwen3TTSPreparedInput.withPreparedInput(validFiles()) { _ -> Void in
            calls += 1
            throw Failure.materializer
        }) { error in
            XCTAssertTrue(error is Failure)
        }
        XCTAssertEqual(calls, 1)
    }

    private func validFiles() -> [String: Data] {
        [
            "config.json": Data(#"{"model_type":"qwen3_tts","tts_model_type":"custom_voice","talker_config":{},"tokenizer_config":{"decode_upsample_rate":123}}"#.utf8),
            "tokenizer_config.json": Data(#"{"tokenizer_class":"Qwen2Tokenizer"}"#.utf8),
            "tokenizer.json": Data(#"{"model":{"type":"BPE","vocab":{"a":1,"b":2,"ab":3},"merges":["a b",["a","b"]]}}"#.utf8),
            "model.safetensors": Data([0x11, 0x22]),
            "speech_tokenizer/config.json": Data(#"{"decoder_config":{},"decode_upsample_rate":1921}"#.utf8),
            "speech_tokenizer/model.safetensors": Data([0x33, 0x44])
        ]
    }

    private func assertRejected(
        _ files: [String: Data], expected: Qwen3TTSPreparedInputError,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        var calls = 0
        XCTAssertThrowsError(try Qwen3TTSPreparedInput.withPreparedInput(files) { _ in
            calls += 1
        }, file: file, line: line) { error in
            XCTAssertEqual(error as? Qwen3TTSPreparedInputError, expected, file: file, line: line)
        }
        XCTAssertEqual(calls, 0, "Invalid input must not enter materialization", file: file, line: line)
    }
}
