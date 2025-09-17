import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property var theme

    property string statusText: "Ready"
    property string logText: "Waiting for job output..."
    property real progress: 0.0

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
                text: statusText
                color: theme.textPrimary
                font.pixelSize: 14
                Layout.fillWidth: true
            }

            ProgressBar {
                value: progress
                from: 0
                to: 1
                Layout.preferredWidth: 220
            }

            Button {
                text: "Cancel"
                enabled: false
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
                    text: logText
                    readOnly: true
                    color: theme.textSecondary
                    background: null
                    wrapMode: Text.Wrap
                }
            }
        }
    }
}
