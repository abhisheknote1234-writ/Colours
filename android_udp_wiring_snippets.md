# Android UDP Wiring Fix (Kotlin Edits)

## `SurveyScreen.kt` (button handlers)
```kotlin
// Send free-text to laptop dashboard
Button(
    onClick = {
        viewModel.sendFreeTextMessage()
    },
    modifier = Modifier
        .fillMaxWidth()
        .height(56.dp),
    enabled = viewModel.currentNoteText.isNotBlank(),
    shape = RoundedCornerShape(16.dp)
) {
    Text("Send Message")
}

// Optional developer test action
IconButton(onClick = { viewModel.runConnectionCheck() }) {
    Icon(Icons.Default.Build, contentDescription = "Test UDP wiring")
}
```

## `EcgViewModel.kt` (exact payload formatting + shared sender hook)
```kotlin
private val udpDebugTag = "UDP_DEBUG"

// Shared UDP dispatch hook (must be set from gateway/service layer)
var onSendUdpReply: ((String) -> Unit)? = null

fun sendFreeTextMessage() {
    val processedText = currentNoteText
        .trim()
        .replace("\n", " ")
        .replace("\r", "")

    if (processedText.isBlank()) {
        Log.w(udpDebugTag, "UDP SEND SKIPPED [free_text]: blank message")
        return
    }

    // STRICT protocol required by Python dashboard
    val payload = "PHONE_MESSAGE:$processedText"

    val sender = onSendUdpReply
    if (sender == null) {
        Log.e(udpDebugTag, "UDP SEND ABORTED [free_text]: sender hook is null")
        return
    }

    Log.d(udpDebugTag, "UDP SEND [free_text] -> $payload")
    sender.invoke(payload) // same hook as scale flow

    // Keep existing UX behavior
    onNoteTextChanged("")
}

fun sendScaleReply(value: Int) {
    // Keep existing SURVEY_REPLY protocol
    val clamped = value.coerceIn(1, 7)
    val payload = "SURVEY_REPLY:$clamped"

    val sender = onSendUdpReply
    if (sender == null) {
        Log.e(udpDebugTag, "UDP SEND ABORTED [scale]: sender hook is null")
        return
    }

    Log.d(udpDebugTag, "UDP SEND [scale] -> $payload")
    sender.invoke(payload) // same hook + same destination via gateway/service
}

fun runConnectionCheck() {
    val sender = onSendUdpReply
    if (sender == null) {
        Log.e(udpDebugTag, "UDP CHECK ABORTED: sender hook is null")
        return
    }

    viewModelScope.launch {
        Log.i(udpDebugTag, "--- STARTING CONNECTION CHECK ---")

        val msgPayload = "PHONE_MESSAGE:hello_from_phone"
        Log.d(udpDebugTag, "UDP SEND [connection_check#1] -> $msgPayload")
        sender.invoke(msgPayload)

        delay(500)

        val scalePayload = "SURVEY_REPLY:7"
        Log.d(udpDebugTag, "UDP SEND [connection_check#2] -> $scalePayload")
        sender.invoke(scalePayload)

        Log.i(udpDebugTag, "--- CONNECTION CHECK PACKETS DISPATCHED ---")
    }
}
```

## `EcgGatewayService.kt` (single destination wiring on UDP:5005)
```kotlin
private val udpDebugTag = "UDP_DEBUG"
private var udpSocket: DatagramSocket? = null

// serverIp must come from app settings (NO hardcoded fallback)
var serverIp: String = ""

fun sendUdpString(message: String, source: String) {
    val targetIp = serverIp.trim()
    if (targetIp.isBlank()) {
        Log.e(udpDebugTag, "UDP SEND ABORTED [$source]: targetIp is blank (check settings)")
        return
    }

    serviceScope.launch(Dispatchers.IO) {
        try {
            if (udpSocket == null || udpSocket?.isClosed == true) {
                udpSocket = DatagramSocket()
            }

            val address = InetAddress.getByName(targetIp)
            val bytes = message.toByteArray(Charsets.UTF_8)
            val packet = DatagramPacket(bytes, bytes.size, address, 5005)

            udpSocket?.send(packet)
            Log.d(udpDebugTag, "UDP SEND [$source] -> $message to $targetIp:5005")
        } catch (e: Exception) {
            Log.e(udpDebugTag, "UDP ERROR [$source]: ${e.message}", e)
        }
    }
}
```

## Hook wiring example (single path for both free-text and scale)
```kotlin
// During ViewModel/service wiring:
viewModel.onSendUdpReply = { payload ->
    val source = when {
        payload.startsWith("PHONE_MESSAGE:") -> "free_text"
        payload.startsWith("SURVEY_REPLY:") -> "scale"
        else -> "other"
    }
    gatewayService.sendUdpString(payload, source)
}
```

## Wire verification checklist (device)
1. In app settings, set laptop IP correctly (same Wi-Fi subnet) and save.
2. Send free text `I am feeling tired` from phone.
   - Android Logcat must show: `UDP SEND [free_text] -> PHONE_MESSAGE:I am feeling tired ...:5005`
   - Python dashboard must show: `PHONE: I am feeling tired`
3. Tap a scale button (e.g., 7).
   - Android Logcat must show: `UDP SEND [scale] -> SURVEY_REPLY:7 ...:5005`
   - Python side should parse as scale reply.
4. Run `runConnectionCheck()` from debug action.
   - Logcat should show both payloads sent in order.
5. If Python still only shows scale values, verify free-text button actually calls `sendFreeTextMessage()` and not scale send path.


## 📝 Wire Verification Checklist (Run on Device)
1. **IP Check:** Open App Settings and verify the **PC IP Address** matches your laptop's current Wi-Fi IP (use `ipconfig` on Windows).
2. **Logcat Monitor:** Open Android Studio **Logcat** and filter by `UDP_DEBUG`.
3. **Free-Text Test:** Type `I am feeling tired` and tap **Send Message**.
   - Verify Logcat prints: `UDP SEND [free_text] -> PHONE_MESSAGE:I am feeling tired to <Your_IP>:5005`
   - Verify Python dashboard chat shows the exact text.
4. **Scale Test:** Tap a scale value (e.g., `4`).
   - Verify Logcat prints: `UDP SEND [scale] -> SURVEY_REPLY:4 to <Your_IP>:5005`
   - Verify Python side receives and parses the scale packet.
5. **Integration Sync:** Run **Connection Check** (or restart session) and confirm both packets arrive on laptop within ~500 ms.
6. **If issue persists:** Confirm the free-text button calls `sendFreeTextMessage()` and not the scale send path.
