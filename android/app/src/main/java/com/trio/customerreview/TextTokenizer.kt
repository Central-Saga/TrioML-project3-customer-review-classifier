package com.trio.customerreview

import android.content.Context
import java.util.Locale

class TextTokenizer(
    context: Context,
    private val vocabFileName: String = "vocab.txt",
    private val maxLength: Int = 128
) {

    private val vocab: Map<String, Int>

    private val unkToken = "[UNK]"
    private val clsToken = "[CLS]"
    private val sepToken = "[SEP]"
    private val padToken = "[PAD]"

    private val unkId: Int
    private val clsId: Int
    private val sepId: Int
    private val padId: Int

    init {
        vocab = loadVocabulary(context)

        unkId = vocab[unkToken]
            ?: error("Token [UNK] tidak ditemukan di vocab.txt")

        clsId = vocab[clsToken]
            ?: error("Token [CLS] tidak ditemukan di vocab.txt")

        sepId = vocab[sepToken]
            ?: error("Token [SEP] tidak ditemukan di vocab.txt")

        padId = vocab[padToken]
            ?: error("Token [PAD] tidak ditemukan di vocab.txt")
    }

    /**
     * Hasil tokenisasi yang langsung bisa diberikan
     * ke OnnxPredictor.
     */
    data class EncodedInput(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray
    )

    /**
     * Tokenisasi satu review menjadi input untuk IndoBERT.
     */
    fun encode(text: String): EncodedInput {

        val normalizedText = normalizeText(text)

        val basicTokens = basicTokenize(normalizedText)

        val wordPieceTokens = mutableListOf<String>()

        for (token in basicTokens) {
            wordPieceTokens.addAll(
                wordPieceTokenize(token)
            )
        }

        // Sisakan ruang untuk [CLS] dan [SEP]
        val availableTokens = maxLength - 2

        val finalTokens = if (
            wordPieceTokens.size > availableTokens
        ) {
            wordPieceTokens.take(availableTokens)
        } else {
            wordPieceTokens
        }

        val tokenIds = mutableListOf<Long>()

        // [CLS]
        tokenIds.add(clsId.toLong())

        // Token hasil WordPiece
        for (token in finalTokens) {
            tokenIds.add(
                vocab[token]?.toLong() ?: unkId.toLong()
            )
        }

        // [SEP]
        tokenIds.add(sepId.toLong())

        // Attention mask:
        // token asli = 1
        // padding = 0
        val attentionMask = MutableList(
            tokenIds.size
        ) {
            1L
        }

        // Padding sampai maxLength
        while (tokenIds.size < maxLength) {
            tokenIds.add(
                padId.toLong()
            )

            attentionMask.add(0L)
        }

        // BERT single sentence => semuanya segment 0
        val tokenTypeIds = LongArray(
            maxLength
        ) { 0L }

        return EncodedInput(
            inputIds = tokenIds.toLongArray(),
            attentionMask = attentionMask.toLongArray(),
            tokenTypeIds = tokenTypeIds
        )
    }

    /**
     * Load vocab.txt.
     *
     * Format:
     * token pada baris 0
     * token pada baris 1
     * dst.
     */
    private fun loadVocabulary(
        context: Context
    ): Map<String, Int> {

        val result = HashMap<String, Int>()

        context.assets
            .open(vocabFileName)
            .bufferedReader(Charsets.UTF_8)
            .useLines { lines ->

                lines.forEachIndexed { index, token ->
                    result[token] = index
                }
            }

        require(result.isNotEmpty()) {
            "vocab.txt kosong atau tidak ditemukan"
        }

        return result
    }

    /**
     * Normalisasi teks dasar.
     *
     * IndoBERT yang kita pakai menggunakan tokenizer
     * model BERT dengan lowercase.
     */
    private fun normalizeText(
        text: String
    ): String {

        return text
            .replace('\u0000', ' ')
            .replace('\uFFFD', ' ')
            .lowercase(Locale.ROOT)
            .trim()
    }

    /**
     * Basic tokenization:
     *
     * - pisahkan whitespace
     * - pisahkan punctuation
     * - pertahankan angka/huruf
     */
    private fun basicTokenize(
        text: String
    ): List<String> {

        val tokens = mutableListOf<String>()

        var currentToken = StringBuilder()

        fun flushCurrentToken() {
            if (currentToken.isNotEmpty()) {
                tokens.add(
                    currentToken.toString()
                )
                currentToken = StringBuilder()
            }
        }

        for (char in text) {

            when {
                char.isWhitespace() -> {
                    flushCurrentToken()
                }

                isPunctuation(char) -> {
                    flushCurrentToken()
                    tokens.add(char.toString())
                }

                else -> {
                    currentToken.append(char)
                }
            }
        }

        flushCurrentToken()

        return tokens
    }

    /**
     * WordPiece tokenizer standar BERT.
     */
    private fun wordPieceTokenize(
        token: String
    ): List<String> {

        if (token.isEmpty()) {
            return emptyList()
        }

        // Token khusus
        if (vocab.containsKey(token)) {
            return listOf(token)
        }

        val chars = token.toCharArray()

        val subTokens = mutableListOf<String>()

        var start = 0

        var isBad = false

        while (start < chars.size) {

            var end = chars.size

            var currentSubToken: String? = null

            while (start < end) {

                val substring =
                    chars.concatToString(
                        start,
                        end
                    )

                val candidate =
                    if (start > 0) {
                        "##$substring"
                    } else {
                        substring
                    }

                if (vocab.containsKey(candidate)) {
                    currentSubToken = candidate
                    break
                }

                end--
            }

            if (currentSubToken == null) {
                isBad = true
                break
            }

            subTokens.add(
                currentSubToken
            )

            start = end
        }

        if (isBad) {
            return listOf(unkToken)
        }

        return subTokens
    }

    /**
     * Punctuation yang perlu dipisahkan
     * seperti tokenizer BERT.
     */
    private fun isPunctuation(
        char: Char
    ): Boolean {

        return when (char) {
            '!', '"', '#', '$', '%', '&',
            '\'', '(', ')', '*', '+',
            ',', '-', '.', '/', ':', ';',
            '<', '=', '>', '?', '@', '[',
            '\\', ']', '^', '_', '`', '{',
            '|', '}', '~' -> true

            else -> {
                val type = Character.getType(char)

                type == Character.CONNECTOR_PUNCTUATION.toInt() ||
                        type == Character.DASH_PUNCTUATION.toInt() ||
                        type == Character.END_PUNCTUATION.toInt() ||
                        type == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
                        type == Character.INITIAL_QUOTE_PUNCTUATION.toInt() ||
                        type == Character.OTHER_PUNCTUATION.toInt() ||
                        type == Character.START_PUNCTUATION.toInt()
            }
        }
    }
}