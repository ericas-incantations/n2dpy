import QtQuick 2.15
import QtQuick.Controls 2.15

QtObject {
    id: theme

    readonly property color background: "#101820"
    readonly property color surface: "#182430"
    readonly property color surfaceAlt: "#1F2D3A"
    readonly property color accent: "#4CC9F0"
    readonly property color accentMuted: "#3A9AD9"
    readonly property color border: "#233445"
    readonly property color textPrimary: "#F5F7FA"
    readonly property color textSecondary: "#B4C5D5"
    readonly property color textMuted: "#7F93A8"

    readonly property int cornerRadius: 12
    readonly property int spacing: 12
    readonly property int padding: 16
}
