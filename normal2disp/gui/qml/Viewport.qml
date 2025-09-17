import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property var theme

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: theme.surface }
            GradientStop { position: 1.0; color: theme.surfaceAlt }
        }
        radius: theme.cornerRadius
        border.color: theme.border
    }

    Label {
        anchors.centerIn: parent
        text: "3D Viewport"
        color: theme.textMuted
        font.pixelSize: 20
    }
}
