import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property var theme
    property var backend

    Rectangle {
        anchors.fill: parent
        color: theme.surface
        border.color: theme.border
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: theme.padding
        spacing: theme.spacing

        RowLayout {
            Layout.fillWidth: true
            spacing: theme.spacing

            Label {
                text: backend ? backend.statusMessage : "Ready"
                color: theme.textPrimary
                font.pixelSize: 14
                Layout.fillWidth: true
            }

            ProgressBar {
                value: backend && backend.inspectRunning ? 0.25 : 0.0
                from: 0
                to: 1
                Layout.preferredWidth: 220
            }

            Button {
                text: "Cancel"
                enabled: backend && backend.inspectRunning
            }

            Button {
                text: "Open Output"
                enabled: false
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            background: Rectangle {
                color: theme.surfaceAlt
                border.color: theme.border
            }

            ScrollView {
                anchors.fill: parent
                TextArea {
                    text: backend ? backend.logText : "Waiting for job output..."
                    readOnly: true
                    color: theme.textSecondary
                    background: null
                    wrapMode: Text.Wrap
                }
            }
        }
    }
}
