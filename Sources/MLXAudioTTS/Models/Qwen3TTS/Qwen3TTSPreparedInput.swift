import CoreFoundation
import Foundation
import Hub
import Tokenizers

enum Qwen3TTSPreparedInputError: Error, Equatable {
    case missingFile(String)
    case invalidModelConfiguration
    case invalidSpeechConfiguration
    case invalidTokenizer
}

struct Qwen3TTSPreparedInput {
    let config: Qwen3TTSModelConfig
    let speechConfig: Qwen3TTSTokenizerConfig
    let tokenizer: PreTrainedTokenizer
    let modelWeights: Data
    let speechWeights: Data

    static func withPreparedInput<Result>(
        _ files: [String: Data],
        materialize: (Qwen3TTSPreparedInput) throws -> Result
    ) throws -> Result {
        func required(_ name: String) throws -> Data {
            guard let data = files[name] else {
                throw Qwen3TTSPreparedInputError.missingFile(name)
            }
            return data
        }

        let configData = try required("config.json")
        let tokenizerConfigData = try required("tokenizer_config.json")
        let tokenizerData = try required("tokenizer.json")
        let modelWeights = try required("model.safetensors")
        let speechConfigData = try required("speech_tokenizer/config.json")
        let speechWeights = try required("speech_tokenizer/model.safetensors")

        let config: Qwen3TTSModelConfig
        do {
            let identity = try JSONDecoder().decode(Config.self, from: configData)
            config = try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: configData)
            guard identity.modelType.string() == "qwen3_tts",
                  identity.ttsModelType.string() == "custom_voice",
                  config.talkerConfig != nil else {
                throw Qwen3TTSPreparedInputError.invalidModelConfiguration
            }
        } catch {
            throw Qwen3TTSPreparedInputError.invalidModelConfiguration
        }

        let speechConfig: Qwen3TTSTokenizerConfig
        do {
            speechConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: speechConfigData)
            guard speechConfig.decoderConfig != nil else {
                throw Qwen3TTSPreparedInputError.invalidSpeechConfiguration
            }
        } catch {
            throw Qwen3TTSPreparedInputError.invalidSpeechConfiguration
        }

        let tokenizer: PreTrainedTokenizer
        do {
            let tokenizerConfig = try Self.tokenizerConfig(from: tokenizerConfigData)
            let preparedTokenizer = try Self.tokenizerConfig(from: tokenizerData)
            // The pinned BPE constructor traps on absent or malformed merges.
            // Reject them before constructing it; preserve valid entries verbatim.
            guard preparedTokenizer.model.vocab.dictionary() != nil,
                  let merges = preparedTokenizer.model.merges.array() else {
                throw Qwen3TTSPreparedInputError.invalidTokenizer
            }
            for merge in merges {
                if let pair = merge.array() {
                    guard pair.count == 2, pair.allSatisfy({ $0.string() != nil }) else {
                        throw Qwen3TTSPreparedInputError.invalidTokenizer
                    }
                } else if let pair = merge.string() {
                    guard pair.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).count == 2 else {
                        throw Qwen3TTSPreparedInputError.invalidTokenizer
                    }
                } else {
                    throw Qwen3TTSPreparedInputError.invalidTokenizer
                }
            }
            // Match the local-folder loader's direct constructor, including config.
            tokenizer = try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: preparedTokenizer)
        } catch {
            throw Qwen3TTSPreparedInputError.invalidTokenizer
        }

        return try materialize(Qwen3TTSPreparedInput(
            config: config, speechConfig: speechConfig, tokenizer: tokenizer,
            modelWeights: modelWeights, speechWeights: speechWeights
        ))
    }

    private static func tokenizerConfig(from data: Data) throws -> Config {
        // NSDictionary retains canonically equivalent, byte-distinct vocabulary keys.
        // A Swift String-keyed dictionary or JSONDecoder would coalesce them.
        guard let object = try JSONSerialization.jsonObject(with: data) as? NSDictionary else {
            throw Qwen3TTSPreparedInputError.invalidTokenizer
        }
        return try configValue(from: object)
    }

    private static func configValue(from object: Any) throws -> Config {
        if let dictionary = object as? NSDictionary {
            var values: [Config.Key: Config] = [:]
            for (key, value) in dictionary {
                guard let key = key as? NSString else {
                    throw Qwen3TTSPreparedInputError.invalidTokenizer
                }
                values[Config.Key(key)] = try configValue(from: value)
            }
            return Config(values)
        }
        if let array = object as? NSArray {
            return try Config(array.map { try configValue(from: $0) })
        }
        if let string = object as? NSString {
            return Config(string as String)
        }
        if let number = object as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return Config(number.boolValue)
            }
            let type = String(cString: number.objCType)
            if type == "f" || type == "d" {
                return Config(number.floatValue)
            }
            if let integer = Int(number.stringValue) {
                return Config(integer)
            }
            return Config(number.floatValue)
        }
        if object is NSNull {
            return try JSONDecoder().decode(Config.self, from: Data("null".utf8))
        }
        throw Qwen3TTSPreparedInputError.invalidTokenizer
    }
}
